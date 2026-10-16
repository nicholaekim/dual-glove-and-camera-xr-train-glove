"""The hand-in folder: layout for all three sets, the professor's text format,
accepted takes only, no images, and missing sessions skipped.

The sessions are the synthetic ones `test_protocol_check.py` builds in the
contract's layout, stills included, so "no photograph goes in" is tested
against a session that has photographs to leave out.
"""
import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

import test_protocol_check as fx
from cam_hand import protocol_check as pc
from cam_hand.prof_format import parse_text
from leap_hand.grasp_validate import SECTION_TITLES as GRASP_SECTIONS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "package_professor_set", REPO / "scripts" / "package_professor_set.py")
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)

LANDMARK_RE = re.compile(r"^(\d+) \((Palm|THUMB|Index|Middle|Ring|Pinky)\): "
                         r"\((-?\d+\.\d\d), (-?\d+\.\d\d), (-?\d+\.\d\d)\)$")


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("sessions")
    grasps = fx.make_grasp_session(root / "a")
    flexion = fx.make_flexion_session(root / "b", [
        fx.FlexTake("index"),
        fx.FlexTake("middle", peaks=[0.3] * 5, accepted=False,
                    reason="span too small"),
        fx.FlexTake("middle")])
    flexion_right = fx.make_flexion_session(root / "b2", [
        fx.FlexTake("ring")], hand="right", camera=False, stills=False)
    sequences = fx.make_sequence_session(root / "c", [
        fx.SeqTake(), fx.SeqTake(glove_override={3: {"ring": 0.45}},
                                 accepted=False, reason="operator redo"),
        fx.SeqTake()])
    # Set B has a current check.csv; Set C has none, so the packager runs
    # the checker itself.
    pc.write_check(pc.check_session(flexion))
    return {"grasps": grasps, "flexion": flexion,
            "flexion_right": flexion_right, "sequences": sequences}


@pytest.fixture(scope="module")
def package(sessions, tmp_path_factory):
    out = tmp_path_factory.mktemp("handin") / "grasp and flexion set for professor 2026-09-28"
    rc = pkg.main(["--out", str(out),
                   "--grasps", str(sessions["grasps"]),
                   "--flexion", str(sessions["flexion"]),
                   str(sessions["flexion_right"]),
                   "--sequences", str(sessions["sequences"]),
                   "--operator", "N Kim"])
    return rc, out


def blocks_of(path: Path):
    text = path.read_text(encoding="utf-8")
    return [b.splitlines() for b in text.strip("\n").split("\n\n")]


def check_block_shape(block):
    assert block[0].startswith("Frame ") and "| Hand ID: " in block[0]
    assert re.match(r"^Wrist: \(-?\d+\.\d\d, -?\d+\.\d\d, -?\d+\.\d\d\)$",
                    block[1]), block[1]
    marks = [LANDMARK_RE.match(line) for line in block[2:]]
    assert len(block) == 23 and all(marks), block
    assert [int(m.group(1)) for m in marks] == list(range(21))


def test_package_runs(package):
    rc, out = package
    assert rc == 0
    assert (out / "README.txt").is_file()


def test_grasps_layout(package, sessions):
    _rc, out = package
    g = out / "grasps"
    names = sorted(p.name for p in g.iterdir())
    expected = set()
    for item in ("cylindrical", "tip_pinch"):
        for n in (1, 2, 3):
            expected |= {f"{item}_left_take{n}.jsonl",
                         f"{item}_left_take{n}_keypoints.txt"}
        expected.add(f"{item}_left_all_frames.txt")
    expected |= {"grasps_summary.csv", "grasp_types_audit.csv"}
    assert set(names) == expected
    # the medoid file is the recorder's own, copied unchanged
    src = sessions["grasps"] / "keypoints"
    (kp,) = [p for p in src.iterdir() if p.name.startswith(
        "cylindrical_left_take1_")]
    assert (g / "cylindrical_left_take1_keypoints.txt").read_text(
        encoding="utf-8") == kp.read_text(encoding="utf-8")
    with open(g / "grasps_summary.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    assert header == pkg.SUMMARY_COLUMNS
    assert len(rows) == 6
    first = rows[0]
    assert (first["grasp_name"], first["hand"], first["take"],
            first["status"]) == ("cylindrical", "left", "1", "accepted")
    assert first["raw_jsonl_file"] == "grasps/cylindrical_left_take1.jsonl"
    assert first["representative_keypoints_file"] == \
        "grasps/cylindrical_left_take1_keypoints.txt"
    assert first["all_frames_file"] == "grasps/cylindrical_left_all_frames.txt"
    assert first["joint_frames_file"] == ""        # no joint frames made
    assert first["tracked_percent"] == "98"
    # no `gate` block in these metas: the take's own count is the gate's
    assert first["reacquisitions_in_static_interval"] == "0"
    assert {r["grasp_label"] for r in rows} == {"cylindrical grasp",
                                                "tip pinch"}
    # times keep their digits, straight from the meta file
    meta = json.loads(next((sessions["grasps"] / "meta").glob(
        "cylindrical_left_take1_*.json")).read_text(encoding="utf-8"))
    assert float(first["static_interval_start"]) == pytest.approx(
        meta["static_interval"][0], abs=1e-6)
    assert float(first["medoid_wall_time"]) == pytest.approx(
        meta["medoid_wall_time"], abs=1e-6)
    assert first["recorder_take_name"].startswith("cylindrical_left_take1_")
    # every file a row names is in the folder
    for r in rows:
        for column in ("raw_jsonl_file", "representative_keypoints_file",
                       "all_frames_file"):
            assert (out / r[column]).is_file(), r[column]


def test_grasp_durations_come_from_timestamps_not_from_line_counts(package):
    """The fixture's takes have both hands in every frame: 180 lines, 90
    frames, 89 frame periods of time. Nothing in the summary is a count of
    lines read as seconds."""
    _rc, out = package
    with open(out / "grasps" / "grasps_summary.csv", encoding="utf-8",
              newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        lines = [json.loads(x) for x in (out / r["raw_jsonl_file"]).read_text(
            encoding="utf-8").splitlines() if x.strip()]
        left = [d["wall_time"] for d in lines if d["hand_side"] == "left"]
        assert r["jsonl_row_count"] == str(len(lines)) == "180"
        assert r["unique_frame_count"] == r["hand_frame_count"] == "90"
        assert float(r["recording_duration_s"]) == pytest.approx(
            max(left) - min(left), abs=0.0006)
        assert float(r["recording_duration_s"]) == pytest.approx(89 / 90.0,
                                                               abs=0.002)
        assert float(r["frame_rate_hz"]) == pytest.approx(90.0, abs=0.5)
        assert r["other_hand_detected"] == "true"
        assert r["other_hand_frame_fraction"] == "1"
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "The number of lines in a .jsonl file is NOT used to calculate " \
           "recording time" in flat
    assert "recording_duration_s = the wall_time of the last frame of the " \
           "labelled hand minus the wall_time of its first frame" in flat
    assert not re.search(r"\d+ frames = \d", text)


def test_grasp_audit_lists_every_grasp_of_the_protocol(package):
    _rc, out = package
    with open(out / "grasps" / "grasp_types_audit.csv", encoding="utf-8",
              newline="") as f:
        rows = {r["grasp_name"]: r for r in csv.DictReader(f)}
    assert list(rows) == ["cylindrical", "tip_pinch"]
    for name, r in rows.items():
        assert (r["accepted_recordings"], r["takes_available"],
                r["target_takes"], r["missing_takes"]) == ("3", "1 2 3", "3",
                                                           "0")
        assert r["representative_keypoints_files"] == "3"
        assert r["all_frames_file"] == f"grasps/{name}_left_all_frames.txt"
        assert r["all_frames_frame_count"] == "270"
        assert r["multi_take_status"] == "COMPLETE"
        # no joint frames were made for this session: one of the four
        # datasets is missing, and the status says so
        assert r["joint_frame_xyz_files"] == "0"
        assert r["static_pose_status"] == "INCOMPLETE"
    assert (rows["tip_pinch"]["attempts_not_kept"],
            rows["tip_pinch"]["open_hand_timeouts"],
            rows["tip_pinch"]["rejected_recordings"]) == ("1", "0", "1")
    assert rows["tip_pinch"]["most_common_reject_reason"] == \
        "fingertip wrong (1 of 1)"
    # three takes each, no joint frames: nothing to compare between takes
    assert rows["tip_pinch"]["between_take_max_spread_deg"] == ""
    assert rows["cylindrical"]["most_common_reject_reason"] == ""


def test_grasp_all_frames_are_camera_millimetres(package):
    _rc, out = package
    path = out / "grasps" / "cylindrical_left_all_frames.txt"
    blocks = blocks_of(path)
    assert len(blocks) == 3 * 90        # three takes, the left hand only
    for b in blocks[:5] + blocks[-5:]:
        check_block_shape(b)
        assert "Hand ID: left" in b[0]
    # the wrist is where the camera measured it, not forced to zero
    wrist = [float(v) for v in re.findall(r"-?\d+\.\d\d", blocks[0][1])]
    assert abs(wrist[1]) > 100.0
    assert len(list(parse_text(path.read_text(encoding="utf-8")))) == 270


def test_all_frames_follow_the_recorders_hand_ids(sessions, tmp_path):
    """A take whose meta names the hand ids the recorder followed exports
    the lines of those ids only, whatever their label; one without them
    keeps the label rule. The summary says how the hand was chosen, which
    ids it has, which label the summary frame had, and, from the lines
    themselves, that another hand was in view."""
    import shutil
    src = tmp_path / "20260928_101500_left"
    shutil.copytree(sessions["grasps"], src)
    (meta_path,) = (src / "meta").glob("cylindrical_left_take1_*.json")
    lines = [json.loads(x) for x in (src / "leap" / (meta_path.stem + ".jsonl"))
             .read_text(encoding="utf-8").splitlines() if x.strip()]
    right_ids = sorted({x["hand_id"] for x in lines if x["hand_side"] == "right"})
    left_ids = sorted({x["hand_id"] for x in lines if x["hand_side"] == "left"})
    assert len(right_ids) == 1 and len(left_ids) == 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # the tracker called the operator's hand "right" in this take: the ids
    # decide, not the label
    meta.update(operator_hand_ids=right_ids, other_hand_ids=left_ids,
                operator_hand_label="right")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(src)]) == 0
    blocks = blocks_of(out / "grasps" / "cylindrical_left_all_frames.txt")
    assert len(blocks) == 3 * 90
    assert all("Hand ID: right" in b[0] for b in blocks[:90])
    assert all("Hand ID: left" in b[0] for b in blocks[90:])
    with open(out / "grasps" / "grasps_summary.csv", encoding="utf-8",
              newline="") as f:
        rows = {r["recorder_take_name"]: r for r in csv.DictReader(f)}
    first = rows[meta_path.stem]
    assert first["hand_selection"] == "followed id"
    assert first["hand_ids"] == str(right_ids[0])
    assert first["tracker_label_on_representative_frame"] == "right"
    assert "representative frame carries the tracker's right label" \
        in first["identity_flag"]
    rest = [r for name, r in rows.items() if name != meta_path.stem]
    assert all(r["hand_selection"] == "tracker label" for r in rest)
    assert all(r["tracker_label_on_representative_frame"] == "left"
               and r["identity_flag"] == "" for r in rest)
    # both hands are in every one of these files, and the summary says so
    # from the lines, whether or not the meta listed the other hand
    assert all(r["other_hand_detected"] == "true" for r in rows.values())
    # section 16 says how the followed hand was told apart as the rule it
    # is (an outside review, 2026-10-07: "is the same hand found again" said
    # more than the recorder knows)
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert max(len(x) for x in readme.splitlines()) <= 96
    s16 = " ".join(readme[readme.rindex("16. OTHER-HAND DETECTION"):
                          readme.rindex("17. POSE TRANSITION TIMING")].split())
    assert ("\"followed id\", 1 recording: the recorder followed the hand it "
            "acquired at OPEN HAND by the tracker's hand_id. A new id appearing "
            "where that hand was, in a frame without it, is treated as the "
            "same hand found again and followed: a rule of id and distance "
            "(within 10 cm, the recorder's FOLLOW_RADIUS_M), not a proof of "
            "identity. An id seen in the same frame, or more than 10 cm away, "
            "is treated as another hand.") in s16
    assert "is the same hand found again" not in " ".join(readme.split())


def test_rejected_takes_and_stills_stay_out(package):
    _rc, out = package
    files = [p for p in out.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix.lower() in
                pkg.IMAGE_EXTENSIONS]
    assert pkg.images_in(out) == []
    names = " ".join(p.name for p in files)
    # the rejected flexion attempt and the rejected sequence attempt
    assert "middle_left_take1_20260928_140001" not in names
    assert "seq4_pairs_left_take2_20260928_140001" not in names
    assert "tip_pinch_left_take2_20260928_140004" not in names


def test_flexion_layout_and_text_export(package, sessions):
    _rc, out = package
    left = out / "finger_flexion" / "left"
    takes = [t for t in pc.load_session(sessions["flexion"]).takes
             if t.accepted]
    assert len(takes) == 2
    for t in takes:
        for name in (f"{t.name}.jsonl", f"camera_{t.name}.jsonl",
                     f"{t.name}.events.jsonl", f"{t.name}.txt"):
            assert (left / name).is_file(), name
        # the JSONL is copied as recorded
        assert (left / f"{t.name}.jsonl").read_bytes() == \
            t.path("glove").read_bytes()
        blocks = blocks_of(left / f"{t.name}.txt")
        n_frames = sum(1 for line in t.path("glove").read_text(
            encoding="utf-8").splitlines() if line.strip())
        assert len(blocks) == n_frames
        for b in blocks[:3] + blocks[-3:]:
            check_block_shape(b)
            # glove text files: wrist at the origin, even though the mock
            # glove streams its wrist 12 cm off it
            assert b[1] == "Wrist: (0.00, 0.00, 0.00)"
    assert (left / "flexion_report.txt").is_file()
    report = (left / "flexion_report.txt").read_text(encoding="utf-8")
    assert "check.csv and check.txt in the session folder" in report
    assert "transfer curve" in report and "lag" in report
    # glove-only right hand: no camera file, still a report
    right = out / "finger_flexion" / "right"
    assert not list(right.glob("camera_*"))
    assert (right / "flexion_report.txt").is_file()
    assert len(list(right.glob("*.events.jsonl"))) == 1


def test_sequences_layout_and_check_csv(package, sessions):
    _rc, out = package
    left = out / "sequences" / "left"
    assert len(list(left.glob("camera_*.jsonl"))) == 2
    assert len(list(left.glob("*.events.jsonl"))) == 2
    with open(left / "sequence_check.csv", encoding="utf-8",
              newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 7                       # accepted takes only
    assert {r["glove_pass"] for r in rows} == {"pass"}
    assert {r["camera_pass"] for r in rows} == {"pass"}
    assert rows[5]["flexed"] == "thumb index middle ring pinky"
    assert float(rows[5]["glove_index"]) > 0.9
    # coupling sits beside pass/fail; the clean fixture has none
    assert {"glove_coupling", "camera_coupling"} <= set(rows[0])
    assert {r["glove_coupling"] for r in rows} == {""}
    # the checker ran in memory: nothing was written into the session
    assert not (sessions["sequences"] / "check.csv").exists()


def test_sequence_check_csv_carries_coupling(tmp_path):
    session = fx.make_sequence_session(tmp_path / "s", [fx.SeqTake(
        glove_override={1: {"middle": 0.45}})], camera=False)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--sequences", str(session)]) == 0
    with open(out / "sequences" / "left" / "sequence_check.csv",
              encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["glove_pass"] == "pass"
    assert re.fullmatch(r"middle 0\.4\d", rows[1]["glove_coupling"])
    assert rows[1]["camera_pass"] == "" and rows[1]["camera_coupling"] == ""


def test_readme_names_the_setup_and_the_conventions(package):
    """One README: the grasp set's twenty sections, then the glove sets."""
    _rc, out = package
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for needle in ("Operator: N Kim", "Sensor: camera only",
                   "gloves on + camera", "Ultraleap Stereo IR 170",
                   "StretchSense", "wrist at the origin",
                   "the Wrist line is the wrist joint's measured position",
                   "KNOWN GLOVE LIMITS",
                   "0.47 s", "60 %", "No photographs",
                   "2026-09-28", "finger_flexion, right hand",
                   "tracker's own", "written as it came",
                   "reacquisitions_in_static_interval",
                   "reported as coupling",
                   "SETS B AND C: THE GLOVE RECORDINGS"):
        assert needle in flat, needle
    assert "MOCK" not in text
    assert "—" not in text
    # the grasp set's part comes first and has its twenty sections in order
    lines = text.splitlines()
    at = -1
    for k, title in enumerate(GRASP_SECTIONS, 1):
        at = lines.index(f"{k}. {title}", at + 1)
    glove = lines.index("SETS B AND C: THE GLOVE RECORDINGS "
                        "(single-finger flexion, sequential movements)")
    assert at < glove
    # the grasp part is wrapped to the page (a path cannot be broken)
    assert max(len(x) for x in lines[:glove] if ":\\" not in x) <= 96


def test_grasp_readme_opening_is_written_from_what_the_folder_holds(
        sessions, tmp_path):
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(sessions["grasps"]),
                     "--operator", "N Kim"]) == 0
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert ("This folder contains Leap Motion / Ultraleap recordings of 2 "
            "grasp types taken from the reference grasp papers. Each grasp "
            "has at least one accepted static-pose recording.") in flat
    # no joint frames in this session: the opening does not claim them
    assert "XYZ joint coordinates, joint-local coordinate frames" not in flat
    assert "are not in this build for every recording" in flat
    assert ("Transition time between different grasp poses is a separate "
            "measurement and is not inferred from these static "
            "recordings.") in flat
    assert "REPRESENTATIVE STATIC HAND CONFIGURATION" in flat
    assert "It is not the last frame and not an average" in flat
    assert "the frame of the static interval nearest its mean hand" in flat
    assert ("Strictly a medoid is the frame nearest all the other frames"
            in flat)
    assert "STATIC-POSE TASK: INCOMPLETE" in flat
    assert "FULL MULTI-TAKE DATA COLLECTION: COMPLETE" in flat
    # a grasp-only folder has no glove part
    assert "SETS B AND C" not in flat and "KNOWN GLOVE LIMITS" not in flat


def add_timed_forming(session: Path, prefix: str, moves, length=2.6,
                      cue_before_first_frame=0.003) -> str:
    """Give one take of a `make_grasp_session` a forming file that can be
    timed: a hand moving as `moves` say (`test_transition.profile`), the cue
    just before its first frame, the hold starting at its end. Returns the
    take's name."""
    import test_transition as tt
    name = fx.add_forming(session, prefix, None)
    rows = tt.profile(moves, length=length)
    for r in rows:
        r["palm_abs"] = r["abs26"][0]
    fx.write_jsonl(session / "forming" / f"{name}.jsonl", rows)
    meta_path = session / "meta" / f"{name}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["coaching"] = {"form_start": tt.T0 - cue_before_first_frame,
                        "acquired": [{"hand_id": 7}], "other_hands": [],
                        "forming_frames": len(rows)}
    meta["take_start"] = tt.T0 + length
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return name


def read_transitions(out: Path):
    with open(out / "transitions" / "pose_transitions.csv", encoding="utf-8",
              newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_package_hands_in_the_forming_files_as_a_transition_dataset(tmp_path):
    """A take's forming file is a recording of a change of pose: it goes to
    transitions\\ as recorded, with a row that times it. A take without one
    has no row, and no static take carries a transition number."""
    session = fx.make_grasp_session(tmp_path / "a")
    timed = add_timed_forming(session, "cylindrical_left_take1",
                              [(0.3, 0.8, 150.0)])
    uncued = fx.add_forming(session, "tip_pinch_left_take3",
                            {"transition_s": None, "forming_frames": 9},
                            frames=9)
    out = tmp_path / "handin"
    assert pkg.main(["--out", str(out), "--grasps", str(session),
                     "--operator", "N Kim"]) == 0
    moves = out / "transitions"
    assert sorted(p.name for p in moves.iterdir()) == [
        "open_hand_to_cylindrical_left_take1.jsonl",
        "open_hand_to_tip_pinch_left_take3.jsonl", "pose_transitions.csv"]
    # copied as recorded
    assert (moves / "open_hand_to_cylindrical_left_take1.jsonl").read_bytes() \
        == (session / "forming" / f"{timed}.jsonl").read_bytes()
    # nothing of it among the static files
    assert not list((out / "grasps").glob("*forming*"))

    header, rows = read_transitions(out)
    assert header == pkg.gh.TRANSITION_COLUMNS
    for column in ("participant_id", "hand", "from_grasp", "to_grasp", "take",
                   "recording_file", "cue_time_s", "movement_start_time_s",
                   "settled_time_s", "transition_duration_s",
                   "movement_start_method", "settled_method", "validity",
                   "notes"):
        assert column in header, column
    by_file = {r["recording_file"]: r for r in rows}
    assert len(rows) == 2
    r = by_file["transitions/open_hand_to_cylindrical_left_take1.jsonl"]
    assert (r["participant_id"], r["hand"], r["from_grasp"], r["to_grasp"],
            r["take"], r["validity"], r["notes"]) == (
        "P01", "left", "open_hand", "cylindrical", "1", "valid", "")
    assert float(r["cue_time_s"]) == pytest.approx(-0.003, abs=0.0006)
    assert float(r["movement_start_time_s"]) == pytest.approx(0.3, abs=0.06)
    assert float(r["settled_time_s"]) == pytest.approx(0.8, abs=0.1)
    # the definition, on the row's own numbers
    assert float(r["transition_duration_s"]) == pytest.approx(
        float(r["settled_time_s"]) - float(r["movement_start_time_s"]),
        abs=0.0011)
    assert float(r["reaction_time_s"]) == pytest.approx(
        float(r["movement_start_time_s"]) - float(r["cue_time_s"]),
        abs=0.0011)
    assert r["static_take_file"] == "grasps/cylindrical_left_take1.jsonl"
    assert r["hand_ids"] == "7"
    assert ("0.1 s or longer, of pose speed at or above 25 mm/s"
            in r["movement_start_method"])
    assert "stays below 10 mm/s for 0.3 s" in r["settled_method"]
    # the pose columns: fifteen joints went 26/15 of 150 mm/s for 0.5 s,
    # and the hand as a whole went nowhere
    assert float(r["pose_change_mm"]) == pytest.approx(98.7, rel=0.1)
    assert float(r["peak_speed_mm_s"]) == pytest.approx(150.0, rel=0.15)
    assert float(r["hand_speed_at_cue_mm_s"]) < 50.0
    # a forming file whose meta has no cue: a row that says so, no numbers
    r = by_file["transitions/open_hand_to_tip_pinch_left_take3.jsonl"]
    assert r["validity"] == "invalid" and "no cue time" in r["notes"]
    assert r["transition_duration_s"] == r["movement_start_time_s"] == ""

    with open(out / "grasps" / "grasps_summary.csv", encoding="utf-8",
              newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        summary = {r["recorder_take_name"]: r for r in reader}
    assert header.index("sensor") == header.index(
        "transition_capture_file") + 1
    # the static table links to the recording of the change and times nothing
    assert not [c for c in header if c.startswith("transition_")
                and c != "transition_capture_file"]
    assert "peak_speed_mm_s" not in header and "last_move_s" not in header
    assert summary[timed]["transition_capture_file"] == \
        "transitions/open_hand_to_cylindrical_left_take1.jsonl"
    assert summary[uncued]["transition_capture_file"] == \
        "transitions/open_hand_to_tip_pinch_left_take3.jsonl"
    old = [r for n, r in summary.items() if n not in (timed, uncued)]
    assert len(old) == 4
    assert all(r["transition_capture_file"] == "" for r in old)
    # camera only: a leap file and no glove file, from the meta or the disk
    assert {r["sensor"] for r in summary.values()} == {"camera"}
    with open(out / "grasps" / "grasp_types_audit.csv", encoding="utf-8",
              newline="") as f:
        audit = {r["grasp_name"]: r for r in csv.DictReader(f)}
    assert (audit["cylindrical"]["transition_captures"],
            audit["cylindrical"]["valid_transitions"]) == ("1", "1")
    assert (audit["tip_pinch"]["transition_captures"],
            audit["tip_pinch"]["valid_transitions"]) == ("1", "0")


def test_package_readme_says_what_transition_timing_is_and_is_not(tmp_path):
    """The section the professor reads: static files are never timed, what
    was recorded, the two moments and their thresholds, and the thresholds
    are the ones the build was run with."""
    session = fx.make_grasp_session(tmp_path / "a")
    add_timed_forming(session, "cylindrical_left_take1", [(0.3, 0.8, 150.0)])
    out = tmp_path / "handin"
    assert pkg.main(["--out", str(out), "--grasps", str(session)]) == 0
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for needle in (
            "17. POSE TRANSITION TIMING",
            "is NOT in the static recordings and is not derived from them",
            "The time between two files, a file's creation time, the gap "
            "between two sessions, the operator's set-up time and rejected "
            "attempts all measure the recording procedure, not the hand",
            "What exists, open hand into a grasp: 1 recording",
            "open_hand to cylindrical, take 1: valid: the pose began to "
            "change",
            "WHAT IS TIMED IS THE POSE, NOT THE HAND OVER THE CAMERA",
            "The length depends on where \"settled\" is put",
            "What does not exist: a recording of the change from one GRASP "
            "type to another grasp type. None has been made.",
            "transition_duration_s = settled_time_s - movement_start_time_s",
            "THE MOVEMENT: the stretch of frames whose pose speed is at or "
            "above 25 mm/s, 0.1 s or longer without a break, that holds the "
            "fastest frame after the cue. MOVEMENT START is its first frame.",
            "SETTLED: after the movement's fastest frame, the first frame "
            "from which the pose speed stays below 10 mm/s for 0.3 s",
            "VALID: the pose was still at the cue",
            "Every number below is a setting of the analysis, not a "
            "constant of it",
            # the joints the hand is laid on are not called rigid, and the
            # build says what leaving the WRIST out of them does
            "they are not a rigid body: the palm itself does not bend "
            "(section 13), yet the WRIST joint shifts against it when the "
            "wrist bends",
            "Which joints the hand is laid on matters little: with the "
            "WRIST left out of the fit, the 1 duration of the table changes "
            "by at most 0.00 s and no validity changes.",
            "lower thresholds take in more of both ends"):
        assert needle in flat, needle
    assert "keeps rigid" not in flat
    assert "—" not in text
    # the thresholds are options, and the README and the table follow them
    out2 = tmp_path / "handin2"
    assert pkg.main(["--out", str(out2), "--grasps", str(session),
                     "--start-speed", "60", "--start-sustain", "0.05",
                     "--settle-speed", "25", "--settle-window", "0.4",
                     "--cue-still", "0.2"]) == 0
    flat = " ".join((out2 / "README.txt").read_text(encoding="utf-8").split())
    assert ("pose speed is at or above 60 mm/s, 0.05 s or longer without a "
            "break") in flat
    assert "the pose speed stays below 25 mm/s for 0.4 s without a break" \
        in flat
    assert "no frame at 60 mm/s or faster within 0.2 s of it" in flat
    # and the two alternatives shown are twice and half of those
    assert "at 120 / 50 mm/s" in flat and "at 30 / 12.5 mm/s" in flat
    _header, rows = read_transitions(out2)
    assert ("0.05 s or longer, of pose speed at or above 60 mm/s"
            in rows[0]["movement_start_method"])
    assert "below 25 mm/s for 0.4 s" in rows[0]["settled_method"]


def test_package_marks_what_cannot_be_timed_and_gives_it_no_duration(tmp_path):
    """Three forming files that are not a clean change: a hand that never
    moved, one that was already moving at the cue, one that settled and
    then adjusted its grip. Only the last has a duration, and it is marked."""
    session = fx.make_grasp_session(tmp_path / "a")
    add_timed_forming(session, "cylindrical_left_take1", [])
    add_timed_forming(session, "cylindrical_left_take2", [(0.0, 0.5, 150.0)])
    add_timed_forming(session, "tip_pinch_left_take3",
                      [(0.3, 0.8, 150.0), (1.6, 1.9, 120.0)], length=3.0)
    out = tmp_path / "handin"
    assert pkg.main(["--out", str(out), "--grasps", str(session)]) == 0
    _header, rows = read_transitions(out)
    got = {(r["to_grasp"], r["take"]): r for r in rows}
    still = got[("cylindrical", "1")]
    assert still["validity"] == "invalid"
    assert still["notes"].startswith("no change of pose after the cue "
                                     "(fastest ")
    assert (still["movement_start_time_s"], still["settled_time_s"],
            still["transition_duration_s"], still["cue_to_settled_s"]) == (
        "", "", "", "")
    early = got[("cylindrical", "2")]
    assert early["validity"] == "invalid"
    assert early["notes"].startswith("the pose was already changing at the "
                                     "cue")
    assert early["transition_duration_s"] == early["reaction_time_s"] == ""
    # the cue and the settle were both recorded: that one time stays
    assert float(early["cue_to_settled_s"]) == pytest.approx(0.5, abs=0.1)
    adjusted = got[("tip_pinch", "3")]
    assert adjusted["validity"] == "questionable"
    assert adjusted["notes"].startswith("settled, then moved again at ")
    assert float(adjusted["transition_duration_s"]) == pytest.approx(0.5,
                                                                   abs=0.12)
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert ("By the rule below, of the 3: 0 valid, 1 questionable "
            "(these have a transition_duration_s), 2 invalid (these have "
            "none).") in flat
    # the files begin at the cue: the README says what that leaves unseen
    assert ("These files begin at the cue, so what the hand did before the "
            "cue is not on record") in flat
    assert "open_hand to cylindrical, take 2: invalid: the pose was already " \
           "changing at the cue" in flat


def _under_the_rules_of_2026_10_07(session: Path, name: str, lost: int) -> None:
    """Make a take's meta one written by today's recorder: recorder_rules,
    the coaching limits, `forming_lost_count`; with `lost`, its forming
    file's every line marked as the recorder marks it."""
    meta_path = session / "meta" / f"{name}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["recorder_rules"] = "2026-10-07"
    meta["coaching"].update({"forming_lost_count": lost, "lost_wait_s": 4.0,
                             "form_window_s": 8.0, "relock_s": 1.0,
                             "reset_limit": 10, "resets": 0})
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if lost:
        path = session / "forming" / f"{name}.jsonl"
        rows = [json.loads(x) for x in path.read_text(
            encoding="utf-8").splitlines() if x.strip()]
        fx.write_jsonl(path, [dict(r, forming_lost_count=lost) for r in rows])


def test_package_says_the_rule_change_where_it_applies(tmp_path, package):
    """A session holding takes recorded before and after the rules of
    2026-10-07, one of them kept after the hand was lost while forming and
    found again, and attempts not kept for the new reasons: section 5 says
    the rules changed, from the metas, how and for how many; the take's
    note is in its summary row and in section 15 under a count line; its
    forming file is counted among the invalid in section 17 with the
    reason and has no duration; section 15's kinds name the new reasons.
    The shared fixture, recorded before, says none of it."""
    from leap_hand import transition as tr

    rp = _recorder()
    session = fx.make_grasp_session(tmp_path / "a")
    old = add_timed_forming(session, "cylindrical_left_take1",
                            [(0.3, 0.8, 150.0)])
    clean = add_timed_forming(session, "cylindrical_left_take2",
                              [(0.3, 0.8, 150.0)])
    lost = add_timed_forming(session, "tip_pinch_left_take1",
                             [(0.3, 0.8, 150.0)])
    _under_the_rules_of_2026_10_07(session, clean, 0)
    _under_the_rules_of_2026_10_07(session, lost, 1)
    loss = {"after_s": 0.6, "height_cm": 25.0, "view_angle_deg": 80.0,
            "cause": "palm turned away", "fix": "turn the palm back"}
    early = {"during": "ready", "after_s": 0.2, "speed_mm_s": 96.0}
    data = json.loads((session / "session.json").read_text(encoding="utf-8"))
    for k, reason in enumerate((
            rp.lost_not_found_reason([loss], "within the 4 s wait for it"),
            rp.reset_reason([early] * 11, [], "closed"))):
        data["takes"].append({"item": "tip_pinch", "take": 4,
                              "name": f"tip_pinch_left_take4_20261007_1200{k}0",
                              "accepted": False, "reason": reason,
                              "decided_at": 1_791_400_000.0 + k, "files": {}})
    (session / "session.json").write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "handin"
    assert pkg.main(["--out", str(out), "--grasps", str(session)]) == 0
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert max(len(x) for x in text.splitlines()) <= 96

    def sec(start: str, end: str) -> str:
        # the section's own heading, not its line in the contents
        return flat[flat.rindex(start):flat.rindex(end)]

    s5 = sec("5. HOW EACH RECORDING WAS CAPTURED", "6. RECORDING DURATION")
    assert ("The recorder's rules changed during these recordings: 2 of the 3 "
            "coached recordings were made under its rules of 2026-10-07 "
            "(recorder_rules in each recording's meta), the rest before "
            "them.") in s5
    assert ("Since that change a hand lost while the grasp was being formed "
            "no longer rejects the attempt") in s5
    assert "(up to 4 s each time; no later than 8 s after the GO beep)" in s5
    assert "1 recording here was kept that way" in s5
    assert "up to 10 times instead of the attempt being dropped" in s5
    note = ("the hand was lost 1 time while forming the grasp and found again "
            "before the hold; the forming file is not timed")
    with open(out / "grasps" / "grasps_summary.csv", encoding="utf-8",
              newline="") as f:
        summary = {r["recorder_take_name"]: r for r in csv.DictReader(f)}
    assert note in summary[lost]["notes"]
    assert note not in summary[clean]["notes"] + summary[old]["notes"]
    s15 = sec("15. QUALITY CONTROL", "16. OTHER-HAND DETECTION")
    assert ("Lost while forming the grasp, found again before the hold "
            "(sections 5, 17): 1 tip_pinch_left_take1: " + note) in s15
    assert f"{pkg.gh.LOST_NOT_FOUND_KIND} 1" in s15
    assert f"{pkg.gh.RESET_LIMIT_KIND} 1" in s15
    s17 = sec("17. POSE TRANSITION TIMING", "18. CURRENT DATASET COMPLETENESS")
    assert ("By the rule below, of the 3: 2 valid, 0 questionable (these have "
            "a transition_duration_s), 1 invalid (these have none); 1 of the "
            "invalid because the hand was lost while forming the grasp") in s17
    assert "2 of them contain a change of pose after the cue; 1 was not read " \
           "at all" in s17
    assert ("open_hand to tip_pinch, take 1: invalid: " + tr.lost_note(1)) in s17
    _header, rows = read_transitions(out)
    got = {(r["to_grasp"], r["take"]): r for r in rows}
    row = got[("tip_pinch", "1")]
    assert row["validity"] == "invalid" and row["notes"] == tr.lost_note(1)
    assert row["transition_duration_s"] == row["movement_start_time_s"] == ""
    assert got[("cylindrical", "2")]["validity"] == "valid"
    # recorded before the rules: none of it
    _rc, full = package
    before = " ".join((full / "README.txt").read_text(encoding="utf-8").split())
    for words in ("Since that change", "recorder_rules", "found again before "
                  "the hold", "lost while forming the grasp"):
        assert words not in before, words


def test_package_reason_kind_groups_closed_before_the_cue():
    assert pkg.reason_kind(
        "closed before the cue 3 times, the last during READY, 0.2 s after "
        "the acquisition beep, at 96 mm/s") == "closed before the cue"
    # a loss and early closes together: the loss names it, as before
    assert pkg.reason_kind(
        "lost the hand 1 time while forming the grasp, the last at 25 cm "
        "(palm turned away: turn it back); closed before the cue 2 times, the "
        "last during READY") == "hand lost while forming the grasp"


def _recorder():
    spec = importlib.util.spec_from_file_location(
        "record_poses_for_package", REPO / "scripts" / "leap" / "record_poses.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    return rp


def test_package_reason_kind_knows_the_reasons_of_the_rules_of_2026_10_07():
    """The recorder's own reason texts since a lost hand is waited for and
    an early close resets the attempt, made by the recorder's functions:
    each has a kind of its own, apart from the earlier rules' kinds."""
    rp = _recorder()
    loss = {"after_s": 0.6, "height_cm": 25.0, "view_angle_deg": 80.0,
            "cause": "palm turned away", "fix": "turn the palm back"}
    early = {"during": "ready", "after_s": 0.2, "speed_mm_s": 96.0}
    for reason in (
            rp.lost_not_found_reason([loss], "within the 4 s wait for it"),
            rp.lost_not_found_reason([loss, loss], "before the forming window "
                                     "ended, 8 s after the GO beep")):
        assert pkg.reason_kind(reason) == pkg.gh.LOST_NOT_FOUND_KIND, reason
    for reason in (rp.reset_reason([early] * 11, [], "closed"),
                   rp.reset_reason([early] * 4, [dict(loss, during="ready")] * 7,
                                   "lost")):
        assert pkg.reason_kind(reason) == pkg.gh.RESET_LIMIT_KIND, reason
    kinds = {pkg.gh.LOST_NOT_FOUND_KIND, pkg.gh.RESET_LIMIT_KIND,
             "hand lost while forming the grasp", "closed before the cue"}
    assert len(kinds) == 4
    assert not any(pkg.gh.is_wait(k) for k in kinds)


def test_package_a_session_without_forming_files_has_an_empty_table(package):
    """The shared fixture, recorded as before 2026-10-02: no recording of a
    change anywhere. The table is there with its header and no row, and the
    README says nothing was recorded."""
    _rc, out = package
    assert sorted(p.name for p in (out / "transitions").iterdir()) == [
        "pose_transitions.csv"]
    header, rows = read_transitions(out)
    assert header == pkg.gh.TRANSITION_COLUMNS and rows == []
    with open(out / "grasps" / "grasps_summary.csv", encoding="utf-8",
              newline="") as f:
        summary = list(csv.DictReader(f))
    assert summary and all(r["sensor"] == "camera"
                           and r["transition_capture_file"] == ""
                           for r in summary)
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert "What exists, open hand into a grasp: nothing." in flat
    assert "The table has its header row only." in flat
    assert "no change of pose was recorded" in flat


def test_package_sensor_reads_the_meta_files():
    ref = pc.TakeRef(name="x_left_take1_20261002_100000", item="x", take=1,
                     accepted=True, reason="", files={})
    assert pkg.take_sensor({"glove": None, "leap": "leap/x.jsonl"}, ref) == "camera"
    assert pkg.take_sensor({"glove": "glove/x.jsonl", "leap": "leap/x.jsonl"},
                           ref) == "glove+camera"
    assert pkg.take_sensor({"glove": "glove/x.jsonl", "leap": None}, ref) == "glove"
    # nothing in the meta and nothing on disk: an empty cell, not an error
    assert pkg.take_sensor(None, ref) == ""


def _short_grasp_session(src: Path, dst: Path) -> Path:
    """A copy of the fixture's grasp session in which tip_pinch is short of
    its takes (its third take rejected, then two attempts with no hand) and
    the item list names a grasp that was never attempted."""
    import shutil
    shutil.copytree(src, dst)
    meta = json.loads((dst / "session.json").read_text(encoding="utf-8"))
    takes = meta["takes"]
    third = [t for t in takes if t["item"] == "tip_pinch" and t["accepted"]][-1]
    third["accepted"] = False
    third["reason"] = (
        "tracked 40 percent: lost 1 time, for 2.0 s (not back by the end) "
        "with the hand at 21 cm (palm turned away: turn the palm back toward "
        "the lenses; tilt it less than about 45 degrees); the gate needs 90 "
        "percent")
    for k in range(2):
        takes.append({"item": "tip_pinch", "take": 3,
                      "name": f"tip_pinch_left_take3_20260928_1450{k}0",
                      "accepted": False,
                      "reason": "no hand was tracked during the take",
                      "decided_at": 1_790_300_000.0 + k, "files": {}})
    meta["items"] = ["cylindrical", "tip_pinch", "hook"]
    meta["takes_per_item"] = 3
    (dst / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return dst


SHORT_ROW_RE = re.compile(r"^  (\S+)\s+(\d+)\s+(\d+)  (.+)$")


def test_readme_lists_grasps_short_of_their_takes(sessions, tmp_path, package):
    short = _short_grasp_session(sessions["grasps"],
                                 tmp_path / "20260928_140000_left")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(short)]) == 0
    text = (out / "README.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    at = lines.index("19. RECORDINGS STILL REQUIRED")
    # the task's own need first, our target second, and never mixed
    assert lines[at + 2] == ("  For the task as set (section 18 A): a first "
                             "recording of hook.")
    assert lines[at + 3] == ("  For our 3-take replication target (section "
                             "18 B), 4 more accepted recordings:")
    assert lines[at + 4].split() == ["grasp_name", "have", "still", "needed",
                                     "takes", "to", "record"]
    rows = {}
    for line in lines[at + 5:]:
        m = SHORT_ROW_RE.match(line)
        if not m:
            break
        rows[m[1]] = (int(m[2]), int(m[3]), m[4])
    # cylindrical has its 3 and is not listed
    assert rows == {"tip_pinch": (2, 1, "3"), "hook": (0, 3, "1 2 3")}
    # under the table: the grasp types whose every recording carries a note
    noted = pkg.gh.all_noted(pkg.gh.collect(pc.load_session(short)))
    assert noted == ["cylindrical", "tip_pinch"]    # the fixture's hand moves
    assert ("Worth making all the same: a recording of "
            + ", ".join(noted) + " that needs no note.") in " ".join(
                text.split())
    worth = next(i for i, x in enumerate(lines)
                 if x.startswith("  Worth making all the same"))
    assert worth > at + 5                           # under the table
    flat = " ".join(text.split())
    # the two questions, answered apart
    assert "STATIC-POSE TASK: INCOMPLETE. 2 of 3 grasp types have an " \
           "accepted recording" in flat
    assert "FULL MULTI-TAKE DATA COLLECTION: INCOMPLETE. 5 of 9 recordings; " \
           "1 of 3 grasp types have their 3; 4 recordings still to make" in flat
    assert "This folder contains Leap Motion / Ultraleap recordings of 2 of " \
           "the 3 grasp types" in flat
    assert "No accepted recording yet for: hook." in flat
    assert "—" not in text
    with open(out / "grasps" / "grasp_types_audit.csv", encoding="utf-8",
              newline="") as f:
        audit = {r["grasp_name"]: r for r in csv.DictReader(f)}
    assert list(audit) == ["cylindrical", "tip_pinch", "hook"]
    assert (audit["tip_pinch"]["accepted_recordings"],
            audit["tip_pinch"]["missing_takes"],
            audit["tip_pinch"]["attempts_not_kept"],
            audit["tip_pinch"]["open_hand_timeouts"],
            audit["tip_pinch"]["rejected_recordings"],
            audit["tip_pinch"]["most_common_reject_reason"]) == (
        "2", "1", "4", "0", "4", "no hand tracked during the take (2 of 4)")
    assert (audit["hook"]["accepted_recordings"],
            audit["hook"]["missing_takes"], audit["hook"]["takes_available"],
            audit["hook"]["all_frames_file"],
            audit["hook"]["most_common_reject_reason"],
            audit["hook"]["static_pose_status"],
            audit["hook"]["multi_take_status"]) == (
        "0", "3", "", "", "not attempted", "INCOMPLETE", "INCOMPLETE")
    assert audit["cylindrical"]["multi_take_status"] == "COMPLETE"
    # a session with every grasp complete says so in one line
    _rc, full = package
    done = (full / "README.txt").read_text(encoding="utf-8")
    assert ("  For the task as set (section 18 A): none. Every grasp type "
            "has its recording.") in done
    assert ("  For our 3-take replication target (section 18 B): none, "
            "every grasp type has its takes.") in done


def test_reject_reasons_are_grouped_by_kind():
    kinds = {
        "no hand was tracked during the take": "no hand tracked during the take",
        "tracked 63 percent: lost 1 time, for 1.9 s (not back by the end) with "
        "the hand at 23 cm (unexplained: none of the measured causes fits: wipe "
        "the lenses); the gate needs 90 percent":
            "hand lost during the take, not back by the end",
        "tracked 53 percent: lost 2 times, longest 1.2 s with the hand at 15 cm "
        "(palm turned away: turn the palm back toward the lenses); the gate "
        "needs 90 percent": "hand lost during the take",
        "tracked 57 percent: no loss inside the take; first tracked 2.2 s after "
        "the start; the gate needs 90 percent":
            "hand first tracked late in the take",
        "no open hand acquired in 30 s": "no open hand acquired in 30 s",
        "lost the hand 3 times while forming the grasp, the last at 25 cm with "
        "the palm 73 degrees from the lens (palm turned away: turn the palm "
        "back toward the lenses)": "hand lost while forming the grasp",
        "operator redo": "operator redo",
        "the operator quit the session at the review": "operator quit at the review",
        "fingertip wrong": "fingertip wrong",
        "the tracker fitted the hand as a right hand for the whole take; the "
        "left hand cannot be measured from that":
            "tracker fitted it as the other hand",
        "the tracker fitted the hand as a right hand for most of the static "
        "interval (36 of 180 frames as left); the left hand cannot be measured "
        "from that": "tracker fitted it as the other hand",
    }
    for reason, kind in kinds.items():
        assert pkg.reason_kind(reason) == kind, reason


def _mocked(src: Path, dst: Path, flags: dict) -> Path:
    """A copy of a session whose session.json carries these mock flags."""
    import shutil
    shutil.copytree(src, dst)
    meta = json.loads((dst / "session.json").read_text(encoding="utf-8"))
    meta.update(flags)
    (dst / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return dst


MOCK_SHAPES = [
    {"mock": True},                                      # Set A, and B/C later
    {"mock": {"glove": True, "glove_follows_cues": True,
              "leap": False}},                          # Sets B and C today
    {"mock": False, "mock_flags": {"glove": False,
                                   "leap": True}},      # Sets B and C later
]


@pytest.mark.parametrize("flags", MOCK_SHAPES)
def test_mock_session_is_refused_unless_allowed(sessions, tmp_path, capsys,
                                                flags):
    mock = _mocked(sessions["flexion"], tmp_path / "20260928_140000_left",
                   flags)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(mock)]) == 1
    text = capsys.readouterr().out
    assert f"mock session, not handed in: {mock}" in text
    assert not out.exists()
    assert pkg.main(["--out", str(out), "--flexion", str(mock),
                     "--allow-mock"]) == 0
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "MOCK DATA" in readme and "It is not a hand-in." in readme
    assert "MOCK session (synthetic data)" in readme
    assert len(list((out / "finger_flexion" / "left").glob(
        "*.events.jsonl"))) == 2


def test_mock_session_beside_real_ones(sessions, tmp_path, capsys):
    mock = _mocked(sessions["grasps"], tmp_path / "grasps_mock",
                   {"mock": True})
    out = tmp_path / "out"
    rc = pkg.main(["--out", str(out), "--grasps", str(mock),
                   "--flexion", str(sessions["flexion"])])
    assert rc == 0
    assert f"mock session, not handed in: {mock}" in capsys.readouterr().out
    assert not (out / "grasps").exists()
    assert (out / "finger_flexion" / "left" / "flexion_report.txt").is_file()
    assert "MOCK" not in (out / "README.txt").read_text(encoding="utf-8")


def test_mock_flags_that_are_all_false_are_a_real_session(sessions, tmp_path):
    real = _mocked(sessions["flexion"], tmp_path / "20260928_140000_left",
                   {"mock": {"glove": False, "leap": False},
                    "mock_flags": {}})
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(real)]) == 0
    assert "MOCK" not in (out / "README.txt").read_text(encoding="utf-8")


def test_missing_session_is_skipped(sessions, tmp_path, capsys):
    out = tmp_path / "out"
    rc = pkg.main(["--out", str(out),
                   "--flexion", str(sessions["flexion"]),
                   "--sequences", str(tmp_path / "not_recorded_yet"),
                   "--grasps", str(tmp_path)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "skipped sequences:" in text and "does not exist" in text
    assert "skipped grasps: no session.json" in text
    assert (out / "finger_flexion" / "left").is_dir()
    assert not (out / "sequences").exists()
    assert "not included in this folder" in (out / "README.txt").read_text(
        encoding="utf-8")
    assert pkg.main(["--out", str(tmp_path / "none"),
                     "--flexion", str(tmp_path / "nope")]) == 1


def test_an_image_in_the_plan_refuses_the_whole_folder(tmp_path, capsys):
    session = fx.make_flexion_session(tmp_path / "s", [fx.FlexTake("index")])
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    # a recorder bug that points the events entry at the take's still
    meta["takes"][0]["files"]["events"] = meta["takes"][0]["files"]["still"]
    (session / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(session)]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert not out.exists()


def test_an_image_left_in_the_folder_fails_the_run(sessions, tmp_path,
                                                   capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "photo.JPG").write_bytes(b"\xff\xd8\xff\xe0 fake")
    assert pkg.main(["--out", str(out), "--flexion",
                     str(sessions["flexion"])]) == 1
    assert "photo.JPG" in capsys.readouterr().out
    # and a renamed image is caught by its first bytes
    renamed = tmp_path / "x.txt"
    renamed.write_bytes(b"\x89PNG\r\n\x1a\n...")
    assert pkg.is_image(renamed)
    assert not pkg.is_image(out.parent / "absent.txt")


def test_a_deep_out_folder_past_max_path(sessions, tmp_path):
    """The hand-in folder sits under a long OneDrive path; a take file name
    is 60 characters on its own. Past 260 characters it must still work."""
    out = tmp_path / ("x" * 110) / ("y" * 110) / "handin"
    assert len(str(out)) > 240
    rc = pkg.main(["--out", str(out), "--flexion", str(sessions["flexion"])])
    assert rc == 0
    left = pkg.os_path(out / "finger_flexion" / "left")
    assert len(list(left.glob("camera_*.jsonl"))) == 2
    assert (left / "flexion_report.txt").is_file()
    # the final scan sees that deep too: an image hidden at a long path fails
    (left / ("z" * 40 + ".png")).write_bytes(b"\x89PNG fake")
    assert len(pkg.images_in(out)) == 1
    assert pkg.main(["--out", str(out), "--flexion",
                     str(sessions["flexion"])]) == 1


# --- joint frames (scripts/joint_frames_view.py --session) ---------------------------

def _with_joint_frames(src: Path, dst: Path) -> Path:
    """A copy of a session with a joint_frames folder as --session leaves it:
    a CSV and a PNG per accepted take (and per camera file for B/C), one PDF,
    plus a CSV of a take that was not accepted."""
    import shutil
    shutil.copytree(src, dst)
    session = pc.load_session(dst)
    out = dst / "joint_frames"
    out.mkdir()
    for t in session.takes:
        stems = [t.name] if session.set == pc.GRASPS else [
            t.name, f"camera_{t.name}"]
        for stem in stems:
            (out / f"{stem}.csv").write_text(
                "joint,parent,x_mm\nWRIST,,0.00\n", encoding="utf-8")
            (out / f"{stem}.png").write_bytes(b"\x89PNG fake drawing")
    (out / "joint_frames.pdf").write_bytes(b"%PDF-1.4 fake\n%%EOF\n")
    return dst


def test_joint_frames_go_in_as_csv_and_pdf_never_png(sessions, tmp_path,
                                                     capsys):
    grasps = _with_joint_frames(sessions["grasps"], tmp_path / "a" / "g")
    flexion = _with_joint_frames(sessions["flexion"], tmp_path / "b" / "f")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--flexion", str(flexion)]) == 0
    text = capsys.readouterr().out
    assert "joint frames 6 CSV and 1 PDF" in text          # 2 grasps x 3 takes
    # Set A: renamed like the take's other hand-in files, and the joint key
    # the grasp README carries written beside them
    g = out / "grasps" / "joint_frames"
    want = {f"{item}_left_take{n}.csv" for item in ("cylindrical", "tip_pinch")
            for n in (1, 2, 3)} | {"joint_frames.pdf", "joint_key.csv",
                                   "joint_key.pdf"}
    assert {p.name for p in g.iterdir()} == want
    assert (g / "joint_frames.pdf").read_bytes() == (
        grasps / "joint_frames" / "joint_frames.pdf").read_bytes()
    # Set B: glove and camera CSVs of the accepted takes, and the PDF
    f = out / "finger_flexion" / "left" / "joint_frames"
    accepted_names = [t.name for t in pc.load_session(flexion).takes
                      if t.accepted]
    assert {p.name for p in f.iterdir()} == {
        n for name in accepted_names
        for n in (f"{name}.csv", f"camera_{name}.csv")} | {"joint_frames.pdf"}
    # the rejected attempt's CSV stays out, and is named
    assert "middle_left_take1_20260928_140001.csv: not an accepted take" in text
    # never a PNG
    assert not list(out.rglob("*.png")) and pkg.images_in(out) == []
    readme = (out / "README.txt").read_text(encoding="utf-8")
    for needle in ("JOINT FRAMES AND THE PAPER'S 24 ANGLES",
                   "joint_frames\\<take>.csv", "<hand>\\joint_frames\\camera_",
                   "joint_frames.pdf", "positive toward the palm",
                   "positive toward the thumb", "Cobos et al. 2009",
                   "MCP_aa", "TMC_fe", "forearm", "millimetres"):
        assert needle in readme, needle
    assert "\u2014" not in readme


def test_a_session_without_joint_frames_still_packages(package):
    rc, out = package
    assert rc == 0
    assert not [p for p in out.rglob("joint_frames*")]
    assert "JOINT FRAMES" not in (out / "README.txt").read_text(
        encoding="utf-8")


def test_an_image_renamed_into_joint_frames_is_still_refused(sessions,
                                                            tmp_path, capsys):
    flexion = _with_joint_frames(sessions["flexion"], tmp_path / "f")
    take = next(t for t in pc.load_session(flexion).takes if t.accepted)
    (flexion / "joint_frames" / f"{take.name}.csv").write_bytes(
        b"\x89PNG a drawing saved under the wrong name")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(flexion)]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert not out.exists()


def test_every_warm_up_of_a_session_is_handed_in(sessions, tmp_path):
    """A resumed session has a warm-up per run and each take names its own:
    every warmup*.json goes in beside the takes, named by session, and the
    README lists each file's open and fist."""
    import shutil
    src = sessions["flexion"]
    dst = tmp_path / "b" / src.name
    shutil.copytree(src, dst)
    warm = (dst / "warmup.json").read_text(encoding="utf-8")
    (dst / "warmup_171012.json").write_text(warm, encoding="utf-8")
    meta = json.loads((dst / "session.json").read_text(encoding="utf-8"))
    meta["warmups"] = ["warmup.json", "warmup_171012.json"]
    meta["takes"][-1]["warmup"] = "warmup_171012.json"
    (dst / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(dst)]) == 0
    left = out / "finger_flexion" / "left"
    assert (left / f"{dst.name}_warmup.json").read_text(
        encoding="utf-8") == warm
    assert (left / f"{dst.name}_warmup_171012.json").is_file()
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "    warm-up warmup.json glove curl, thumb index" in readme
    assert "    warm-up warmup_171012.json glove curl, thumb index" in readme
    assert "<session>_warmup*.json" in readme
    assert "\u2014" not in readme


def test_one_warm_up_reads_as_before(package):
    _rc, out = package
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "    warm-up glove curl, thumb index middle ring pinky: open " \
        in readme
    assert list((out / "finger_flexion" / "left").glob("*_warmup.json"))


# --- a rebuild destroys nothing ---------------------------------------------------------

def test_the_build_time_in_the_readme_is_the_last_build_that_changed_a_file(
        sessions, tmp_path, capsys, monkeypatch):
    """A rebuild that changes nothing leaves the README and its time alone;
    one that changes any file writes its own time, even when the README's
    text is otherwise the same."""
    import time as _time
    real = _time.strftime
    now = {"minute": "2031-01-01 00:00"}

    def clock(fmt, *args):
        return now["minute"] if fmt == "%Y-%m-%d %H:%M" else real(fmt, *args)
    monkeypatch.setattr(pkg.time, "strftime", clock)

    def built(folder):
        (line,) = [x for x in (folder / "README.txt").read_text(
            encoding="utf-8").splitlines() if x.startswith("Folder built:")]
        return line

    out = tmp_path / "handin"
    base = ["--out", str(out), "--grasps", str(sessions["grasps"])]
    assert pkg.main(base + ["--superseded", str(tmp_path / "a1")]) == 0
    assert "2031-01-01 00:00" in built(out)
    # an hour later, nothing to change: the time stays
    now["minute"] = "2031-01-01 01:00"
    assert pkg.main(base + ["--superseded", str(tmp_path / "a2")]) == 0
    assert "2031-01-01 00:00" in built(out)
    assert not (tmp_path / "a2").exists()
    # a derived file is gone: the build writes it again, and says when
    victim = sorted((out / "grasps").glob("*_keypoints.txt"))[0]
    victim.unlink()
    capsys.readouterr()
    assert pkg.main(base + ["--superseded", str(tmp_path / "a3")]) == 0
    said = capsys.readouterr().out
    assert victim.is_file() and ": 1 new, 1 changed, " in said
    assert "2031-01-01 01:00" in built(out)
    # the README it replaced is kept, with the earlier time
    assert "2031-01-01 00:00" in built(tmp_path / "a3")



def _digests(folder: Path) -> dict:
    import hashlib
    return {p.relative_to(folder).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_a_rebuild_sets_aside_what_it_changes_and_deletes_nothing(
        sessions, tmp_path, capsys):
    before = _digests(sessions["grasps"])
    out = tmp_path / "handin"
    base = ["--out", str(out), "--grasps", str(sessions["grasps"])]
    assert pkg.main(base + ["--superseded", str(tmp_path / "aside1")]) == 0
    first = capsys.readouterr().out
    assert " new, 0 changed, 0 unchanged" in first
    assert not (tmp_path / "aside1").exists()     # nothing was there before

    # the same build again: not one file is touched, the README included
    stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    built = [x for x in (out / "README.txt").read_text(
        encoding="utf-8").splitlines() if x.startswith("Folder built:")]
    assert len(built) == 1
    assert pkg.main(base + ["--superseded", str(tmp_path / "aside2")]) == 0
    second = capsys.readouterr().out
    assert ": 0 new, 0 changed, " in second
    assert not (tmp_path / "aside2").exists()
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*")
            if p.is_file()} == stamps

    # a table edited by hand, a file an earlier build made and this one
    # does not, and a note someone put at the top of the folder
    summary = out / "grasps" / "grasps_summary.csv"
    good = summary.read_bytes()
    summary.write_bytes(good + b"edited by hand\n")
    stale = out / "grasps" / "old_name_left_take9.jsonl"
    stale.write_text("{}\n", encoding="utf-8")
    note = out / "notes from the professor.md"
    note.write_text("keep me", encoding="utf-8")
    aside = tmp_path / "aside3"
    assert pkg.main(base + ["--superseded", str(aside)]) == 0
    third = capsys.readouterr().out
    assert summary.read_bytes() == good                    # rebuilt
    assert (aside / "grasps" / "grasps_summary.csv").read_bytes() \
        == good + b"edited by hand\n"                      # the edit is kept
    assert not stale.exists()
    assert (aside / "grasps" / "old_name_left_take9.jsonl").read_text(
        encoding="utf-8") == "{}\n"
    assert note.read_text(encoding="utf-8") == "keep me"
    assert "1 replaced and 1 no longer made" in third
    assert "left in place (not made by this build): notes from the " \
           "professor.md" in third
    assert not list(out.rglob("*.building"))
    # and the session folder was only read
    assert _digests(sessions["grasps"]) == before


def test_superseded_files_go_beside_the_folder_by_default(sessions, tmp_path):
    out = tmp_path / "deliver" / "grasp set"
    base = ["--out", str(out), "--grasps", str(sessions["grasps"])]
    assert pkg.main(base) == 0
    assert not (out.parent / "_superseded").exists()
    (out / "grasps" / "grasps_summary.csv").write_text("x", encoding="utf-8")
    assert pkg.main(base) == 0
    (stamp,) = (out.parent / "_superseded" / "grasp set").iterdir()
    assert re.fullmatch(r"\d{8}_\d{6}", stamp.name)
    assert (stamp / "grasps" / "grasps_summary.csv").read_text(
        encoding="utf-8") == "x"
    # nothing of it inside the hand-in folder
    assert not [p for p in out.rglob("*") if "_superseded" in p.parts]


# --- the joint frames and the 24 angles, made for real ------------------------------------

def _view():
    spec = importlib.util.spec_from_file_location(
        "joint_frames_view_p", REPO / "scripts" / "joint_frames_view.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grasp_set_with_joint_frames_has_the_24_angles_and_is_ready(
        sessions, tmp_path, capsys):
    """`joint_frames_view --session` run on the session first, as the
    rebuild does: every take then has its joint-frame table, the 24 angles
    are in one table, and each grasp is READY."""
    import shutil
    from xr_hand import joint_frames as jf
    src = tmp_path / "a" / sessions["grasps"].name
    shutil.copytree(sessions["grasps"], src)
    _view().export_session(src)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(src)]) == 0
    printed = capsys.readouterr().out
    assert "joint frames 6 CSV and 1 PDF" in printed and "PROBLEM" not in printed
    with open(out / "grasps" / "joint_angles_24.csv", encoding="utf-8",
              newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    assert header == pkg.gh.ANGLE_COLUMNS
    assert header[5:29] == jf.DOF24_NAMES and len(rows) == 6
    for r in rows:
        with open(out / r["joint_frames_file"], encoding="utf-8",
                  newline="") as f:
            table = {x["joint"]: x for x in csv.DictReader(f)}
        assert len(table) == 26
        # each of the 18 given is the cell of the table the README says
        # it is; the six this tracker does not give are empty, and their
        # numbers stand under names that say what they are
        gh = pkg.gh
        assert len(gh.GIVEN_NAMES) == 18 and len(gh.NOT_GIVEN) == 6
        for name in gh.GIVEN_NAMES:
            joint, column = jf.DOF24_SOURCE[name]
            assert r[name] == table[joint][column], name
        for name in gh.NOT_GIVEN:
            assert r[name] == "", name
        assert r["T_metacarpal_flex_wrist_ref"] == \
            table["THUMB_METACARPAL"]["flex_deg"]
        assert r["T_metacarpal_abd_wrist_ref"] == \
            table["THUMB_METACARPAL"]["abd_deg"]
        assert r["L_metacarpal_flex_wrist_ref"] == \
            table["LITTLE_METACARPAL"]["flex_deg"]
        extra = [float(r[n]) for n in list(gh.METACARPAL_COLUMNS)
                 + jf.MCP_PAPER_NAMES + ["forearm_to_palm_deg"]]
        assert all(abs(v) <= 360.0 for v in extra)
        # the two orders agree on a joint that is not abducted, and the
        # paper's order never reads a flexion as more than the bone's turn
        for letter in "IMRL":
            fe, aa = float(r[f"{letter}_MCP_fe"]), float(r[f"{letter}_MCP_aa"])
            fe_p = float(r[f"{letter}_MCP_fe_paper_order"])
            import math
            want = math.degrees(math.asin(
                math.sin(math.radians(fe)) * math.cos(math.radians(aa))))
            assert fe_p == pytest.approx(want, abs=0.02)
    with open(out / "grasps" / "grasp_types_audit.csv", encoding="utf-8",
              newline="") as f:
        audit = list(csv.DictReader(f))
    assert all(a["static_pose_status"] == "READY"
               and a["joint_frame_xyz_files"] == "3"
               and a["joint_angle_rows"] == "3" for a in audit)
    with open(out / "grasps" / "grasps_summary.csv", encoding="utf-8",
              newline="") as f:
        summary = list(csv.DictReader(f))
    assert all((out / r["joint_frames_file"]).is_file() for r in summary)
    assert (out / "grasps" / "joint_frames" / "joint_frames.pdf").is_file()
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert ("The package includes raw tracked joint data, representative "
            "static hand keypoints, complete frame-level pose data, XYZ "
            "joint coordinates, joint-local coordinate frames, and derived "
            "joint angles.") in flat
    assert "STATIC-POSE TASK: COMPLETE. 2 of 2 grasp types have an accepted " \
           "recording; 2 of 2 are READY" in flat
    for needle in ("b = asin(R13)", "a = atan2(-R23, R33)",
                   "c = atan2(-R12, R11)", "flex_deg = -a x 180/pi",
                   "abd_deg = s x b x 180/pi", "twist_deg = -s x c x 180/pi",
                   "s = +1 for a right hand and -1 for a left hand",
                   "sin(fe_paper) = sin(fe) x cos(aa)",
                   "tan(aa_paper) = tan(aa) / cos(fe)",
                   "WHERE THESE NUMBERS ARE NOT THE PAPER'S",
                   "+X along the module's long side, to the operator's right",
                   "+Y straight up out of the module, toward the hand",
                   "+Z horizontal, toward the operator",
                   "R_wrist^T (p - p_wrist)"):
        assert needle in flat, needle


def test_a_joint_frame_table_of_another_frame_is_named_as_a_problem(
        sessions, tmp_path, capsys):
    """The table handed in must be the representative frame's: one that is
    not (an export from before a retake, a file edited by hand) is said."""
    grasps = _with_joint_frames(sessions["grasps"], tmp_path / "g")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps)]) == 0
    printed = capsys.readouterr().out
    assert printed.count("PROBLEM: ") == 6
    assert "is not the table of the take's representative frame" in printed
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert "Found wrong or missing by this build:" in flat


# --- what the README says that the build worked out (review of 2026-10-05) ------------
def test_every_summary_column_is_explained_and_none_is_explained_twice():
    gh = pkg.gh
    assert set(gh.COLUMN_HELP) == set(gh.SUMMARY_COLUMNS)
    lines = gh.column_help(gh.SUMMARY_COLUMNS)
    starts = [x[4:].split(": ")[0] for x in lines if x.startswith("    ")
              and not x.startswith("     ")]
    assert starts == gh.SUMMARY_COLUMNS
    assert max(len(x) for x in lines) <= gh.WIDTH
    # a value the build copies from the recorder says so, and no other does
    copied = {c for c in gh.SUMMARY_COLUMNS if "copied" in gh.COLUMN_HELP[c]}
    assert {"recorder_window_s", "tracked_percent", "decided_by",
            "grab_strength", "curl_thumb", "palm_height_cm",
            "view_angle_deg"} <= copied
    assert not copied & {"recording_duration_s", "unique_frame_count",
                         "static_interval_mean_speed_mm_s",
                         "index_metacarpal_mm", "other_hand_detected"}
    # the curl reads the way its sign says
    assert "SMALLER = more curled" in gh.COLUMN_HELP["curl_thumb"]


def test_the_readme_examples_are_what_the_code_gives():
    import math
    gh = pkg.gh
    fe, aa = gh.paper_order_example(60.0, 10.0)
    assert math.sin(math.radians(fe)) == pytest.approx(
        math.sin(math.radians(60.0)) * math.cos(math.radians(10.0)))
    assert math.tan(math.radians(aa)) == pytest.approx(
        math.tan(math.radians(10.0)) / math.cos(math.radians(60.0)))
    assert (round(fe, 1), round(aa, 1)) == (58.5, 19.4)
    # near 90 of flexion a small lean is a large abduction in that order
    assert gh.paper_order_example(85.0, 2.0)[1] == pytest.approx(21.8, abs=0.05)
    assert gh.paper_order_example(30.0, 0.0) == pytest.approx((30.0, 0.0))
    text = gh.known_move_text({})
    for needle in ("in exactly 0.50 s", "0.34 s for a change of 15 mm RMS",
                   "0.43 s for a change of 30 mm RMS",
                   "0.49 s for a change of 61 mm RMS",
                   "0.51 s for a change of 122 mm RMS",
                   "A change of 8 mm RMS is not seen as a movement at all",
                   "Carrying the whole hand 150 mm across the camera during "
                   "the change alters none of these readings"):
        assert needle in text, needle
    # other thresholds, another sentence: nothing in it is typed
    assert gh.known_move_text({"start_speed_mm_s": 15.0,
                               "settle_speed_mm_s": 5.0}) != text
    assert (gh._between(7.1, 7.3), gh._between(7.0, 19.4)) == ("7", "7 to 19")
    assert gh._between(0.24, 0.26, 1) == "0.2 to 0.3"


def test_the_sources_are_the_protocols_and_the_notes_beside_it(tmp_path):
    gh = pkg.gh
    items = [{"id": "a", "same_as": ["b", "c"]}, {"id": "b", "same_as": ["a"]},
             {"id": "c", "same_as": ["a"]}, {"id": "d", "same_as": []},
             {"id": "e", "same_as": ["f"]}, {"id": "f"}, {"id": "g"}]
    # a, b, c are one configuration; e and f another; d and g alone
    assert gh.configurations(items) == (4, 4)
    assert gh.configurations([{"id": "x"}, {"id": "y"}]) == (0, 2)

    class Fake:
        def __init__(self, protocol_file, description=""):
            self.meta = {"protocol_file": str(protocol_file)}
            self.protocol = {"description": description}
    protocol = tmp_path / "grasps.json"
    protocol.write_text("{}", encoding="utf-8")
    assert gh.source_notes(Fake(protocol)) == []
    (tmp_path / "grasps_source_notes.txt").write_text(
        "First   note,\twith  spaces.\n\n\nSecond note.\n", encoding="utf-8")
    assert gh.source_notes(Fake(protocol)) == ["First note, with spaces.",
                                               "Second note."]
    assert gh.source_notes(Fake("")) == []
    cited = gh.citations(Fake(protocol, (
        "Set A. Labels are the papers' own terms. p1 = Heumer, 'A title', "
        "2007 (Figure 1 and Table 1). p2 = de Souza et al., 'Another', "
        "74 (2015) 108-121 (Table 1 and Fig. 13, ten scenarios). The "
        "'same_as' field links grasps.")))
    assert len(cited) == 2
    assert cited[0].startswith("p1 = Heumer") and cited[0].endswith("Table 1).")
    assert cited[1].startswith("p2 = de Souza") and cited[1].endswith(
        "scenarios).")
    # the repository's own notes for the real protocol are found the same way
    real = gh.source_notes(Fake("protocols/grasps.json"))
    assert len(real) == 3 and all("our reading" in n or "drawing" in n
                                  for n in real)


def test_a_close_made_after_a_lost_one_is_said_to_be_the_slower_close():
    gh = pkg.gh
    assert gh.forming_context({}) == ""
    assert gh.forming_context({"coaching": {"lost_while_forming": 0}}) == ""
    one = gh.forming_context({"coaching": {"lost_while_forming": 1}})
    two = gh.forming_context({"coaching": {"lost_while_forming": 2}})
    assert "1 earlier close of this attempt" in one
    assert "2 earlier closes of this attempt" in two
    assert one.endswith("this is the slower close")
    # under the rules of 2026-10-07 nobody is told to close slower: a lost
    # hand is waited for, and its forming file says itself why it is not timed
    assert gh.forming_context({"coaching": {"lost_while_forming": 1,
                                            "forming_lost_count": 1}}) == ""


def test_a_grasp_whose_every_recording_carries_a_note_is_named():
    from types import SimpleNamespace as NS
    gh = pkg.gh

    def take(item, noted):
        return NS(item=item, n=1, facts=NS(abnormal=noted))
    takes = [take("a", True), take("a", True), take("b", True),
             take("b", False), take("c", False)]
    gs = NS(item_ids=["a", "b", "c", "d"],
            of=lambda i: [t for t in takes if t.item == i])
    # a: both noted; b: one clean; c: clean; d: no recording at all
    assert gh.all_noted(gs) == ["a"]
    assert "recordings_without_a_note" in gh.AUDIT_COLUMNS


def test_the_takes_of_one_grasp_are_compared_and_the_links_go_both_ways():
    from types import SimpleNamespace as NS
    gh = pkg.gh

    def take(**angles):
        full = {n: 10.0 for n in gh.GIVEN_NAMES}
        full.update(angles)
        return NS(angles=full, joint_frames="a table")
    a, b, c = take(I_PIP=20.0), take(I_PIP=75.0, T_IP=30.0), take(I_PIP=50.0)
    worst, name, spreads = gh.between_take_spread([a, b, c])
    assert (worst, name) == (55.0, "I_PIP")
    assert spreads["T_IP"] == 20.0 and spreads["M_PIP"] == 0.0
    assert gh.between_take_spread([a]) is None
    # a take without its joint-frame table has no row in the angle table,
    # and is not compared
    assert gh.between_take_spread([a, NS(angles=None, joint_frames="x")]) is None
    assert gh.between_take_spread([a, NS(angles=b.angles, joint_frames=None)]) is None
    # two angles 55 and 20 apart, sixteen the same: the RMS over the 18
    assert gh.angle_rms(a, b) == pytest.approx(
        ((55.0 ** 2 + 20.0 ** 2) / 18) ** 0.5)
    # the protocol links some grasps from one side only: followed both ways
    items = [{"id": "a", "same_as": ["b", "c"]}, {"id": "b", "same_as": ["a"]},
             {"id": "c"}, {"id": "d"}]
    assert gh.same_configuration(items) == {"a": ["b", "c"], "b": ["a", "c"],
                                            "c": ["a", "b"], "d": []}
    # a wait for an open hand is not a rejected recording
    assert gh.is_wait("no open hand acquired in 30 s")
    assert not gh.is_wait("hand lost during the take")
    kinds = {"no open hand acquired in 30 s": 6, "hand lost during the take": 1}
    assert gh.most_common_reject(kinds, True) == \
        "hand lost during the take (1 of 1)"
    assert gh.most_common_reject({"no open hand acquired in 30 s": 6},
                                 False) == ""
    assert gh.most_common_reject({}, False) == "not attempted"


def test_a_rigid_palm_is_told_from_the_takes_and_only_then():
    """The four CMC angles against the palm joint: the same in every take
    means the tracker's palm does not bend, and the README then says the
    four are not measured; a palm that does bend is not called rigid."""
    from types import SimpleNamespace as NS
    gh = pkg.gh
    names = [n for n in gh.jf.PALM_REF_NAMES if "_CMC_" in n]
    assert len(names) == 4

    def gs(rows):
        return NS(takes=[NS(angles=dict(zip(names, r))) for r in rows])
    same = gh.rigid_palm(gs([(8.65, 8.53, 8.59, 6.29)] * 3
                            + [(8.66, 8.53, 8.58, 6.29)]))
    assert same == pytest.approx(dict(zip(names, (8.65, 8.53, 8.59, 6.29))),
                                 abs=0.011)
    assert gh.rigid_palm(gs([(8.65, 8.53, 8.59, 6.29),
                             (9.65, 8.53, 8.59, 6.29)])) is None
    assert gh.rigid_palm(gs([(8.65, 8.53, 8.59, 6.29)])) is None   # one take
    assert gh.rigid_palm(NS(takes=[NS(angles=None)])) is None


def test_the_readme_says_what_was_judged_by_eye_and_what_was_not_measured(
        sessions, tmp_path):
    import shutil
    grasps = tmp_path / "a" / sessions["grasps"].name
    shutil.copytree(sessions["grasps"], grasps)
    _view().export_session(grasps)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--operator", "N Kim"]) == 0
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for needle in (
            "judged, by eye against the paper's picture, to have made",
            "No measurement in this folder checks a hand shape against a "
            "paper",
            "The gate is about tracking. It sets no limit on how much the "
            "hand moved",
            "Columns of grasps_summary.csv, in order.",
            "HOW STILL IT WAS. \"Stillest\" is not \"still\"",
            "are limits for a note in this README, not rules a recording "
            "was accepted by",
            "That is the file's resolution, not the tracker's accuracy, "
            "which was not measured in this work",
            "SCALE: MILLIMETRES ARE NOT COMPARABLE BETWEEN RECORDINGS AS "
            "THEY STAND",
            "tan(twist / 2) = -tan(flex / 2) x tan(abd / 2)",
            "The thumb's two TMC angles are not given",
            "NOT GIVEN (item 2); the columns are empty",
            "18 filled, 6 empty",
            "NOT EVERY FRAME IS THE POSE",
            "AGREEMENT BETWEEN THE TAKES OF ONE GRASP",
            "Its axes are the forearm's, as the tracker estimates them",
            "No accuracy figure for these angles was measured in this work",
            "What is NOT checked by anything: that a hand shape is the "
            "grasp its label names",
            "sharing nothing with the code that made the tables",
            "Three different things, with three different answers.",
            "C. THE TIME TO CHANGE POSE",
            "TRANSITION TIMING: MISSING. No change of pose has been "
            "recorded",
            "For the task as set (section 18 A):",
            "Every file listed in section 4 is written by one command"):
        assert needle in flat, needle
    for column in pkg.gh.SUMMARY_COLUMNS:
        assert f"\n    {column}: " in text, column
    assert max(len(x) for x in text.splitlines() if ":\\" not in x
               and ":/" not in x) <= 96


# --- the pictures: each take's still cut to the recorded hand ---------------------
# A still as the viewer writes one: dim infrared, a bright hand with the
# tracker's skeleton drawn in the viewer's colour, and a bright "face" far
# from the hand that the cut must take away.
STILL_W, STILL_H = 260, 300
FACE_GREY, HAND_GREY = 200, 215
CYAN, RED = (0, 220, 255), (255, 60, 60)


def draw_still(path: Path, hand: str = "left", skeleton: bool = True,
               seed: int = 0) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    rgb = (np.full((STILL_H, STILL_W, 3), 40, np.uint8)
           + rng.integers(0, 12, (STILL_H, STILL_W, 1), dtype=np.uint8))
    cv2.circle(rgb, (215, 40), 30, (FACE_GREY,) * 3, -1)
    grey = (HAND_GREY,) * 3
    palm, wrist = (110, 200), (110, 295)
    tips = [(55, 150), (75, 105), (105, 95), (135, 100), (160, 115)]
    cv2.ellipse(rgb, palm, (42, 48), 0, 0, 360, grey, -1)
    cv2.rectangle(rgb, (95, 230), (125, STILL_H - 1), grey, -1)
    for tip in tips:
        cv2.line(rgb, palm, tip, grey, 16)
    if skeleton:
        colour = CYAN if hand == "left" else RED
        cv2.line(rgb, wrist, palm, colour, 2)
        for tip in tips:
            cv2.line(rgb, palm, tip, colour, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="PNG")


def _with_pictures(src: Path, dst: Path, without=()) -> Path:
    """A copy of a grasp session whose stills show the hand (the fixture's
    are not pictures); the takes named in `without` get a still with no
    skeleton drawn on it."""
    import shutil
    shutil.copytree(src, dst)
    for k, t in enumerate(pc.load_session(dst).takes):
        still = t.files.get("still")
        if still is not None and t.accepted:
            draw_still(still, skeleton=f"{t.item}_take{t.take}" not in without,
                       seed=k)
    return dst


STEMS = [f"{item}_left_take{n}" for item in ("cylindrical", "tip_pinch")
         for n in (1, 2, 3)]


def _summary(out: Path):
    with open(out / "grasps" / "grasps_summary.csv", encoding="utf-8",
              newline="") as f:
        return list(csv.DictReader(f))


def _relative_images(out: Path):
    root = Path(pkg.plain(pkg.os_path(out)))
    return {Path(pkg.plain(p)).relative_to(root).as_posix()
            for p in pkg.images_in(out)}


def test_the_pictures_are_the_stills_cut_to_the_hand_and_the_only_images(
        sessions, tmp_path, capsys):
    import numpy as np
    from PIL import Image
    from pypdf import PdfReader

    grasps = _with_pictures(sessions["grasps"], tmp_path / "a" / "g")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--operator", "N Kim"]) == 0
    assert "6 picture(s)" in capsys.readouterr().out
    images = out / "grasps" / "images"
    assert {p.name for p in images.iterdir()} == (
        {f"{s}.png" for s in STEMS} | {"grasp_images.pdf"})
    # the planned pictures are every image of the folder, and nothing else
    assert _relative_images(out) == {f"grasps/images/{s}.png" for s in STEMS}
    for s in STEMS:
        rgb = np.asarray(Image.open(images / f"{s}.png").convert("RGB"))
        assert rgb.shape[0] < STILL_H and rgb.shape[1] < STILL_W
        assert not (rgb == FACE_GREY).all(axis=2).any()          # face cut away
        assert (rgb == HAND_GREY).all(axis=2).sum() > 2000       # the hand kept
        assert ((rgb[..., 0] < 90) & (rgb[..., 1] > 150)
                & (rgb[..., 2] > 190)).sum() > 150               # its skeleton
    # the summary names each, the audit counts them, the PDF has a page per
    # grasp type, and the README says what they are
    rows = _summary(out)
    assert [r["image_file"] for r in rows] == [
        f"grasps/images/{s}.png" for s in STEMS]
    with open(out / "grasps" / "grasp_types_audit.csv", encoding="utf-8",
              newline="") as f:
        assert [r["recordings_with_an_image"] for r in csv.DictReader(f)] == [
            "3", "3"]
    assert len(PdfReader(str(images / "grasp_images.pdf")).pages) == 2
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert ("Images: one infrared picture per recording, cut to the recorded "
            "hand, grasps\\images (section 5). No other photographs are "
            "included.") in flat
    assert "No photographs are included." not in flat
    for needle in ("Images of the grasp types",
                   "Which joint each angle corresponds to",
                   "THE PICTURES. One infrared frame per recording",
                   "1. The skeleton: the pixels whose R, G and B (0 to 255) "
                   "lie in R 0 to 89, G 151 to 255, B 191 to 255",
                   "the viewer's cyan (0, 220, 255)",
                   "3. Kept: a pixel within 0.3 radii of the nearest "
                   "skeleton pixel, or within 0.5 radii and at least 0.7 "
                   "times as bright as the hand",
                   "within 6 pixels of the skeleton",
                   "(R 201 to 255, G 201 to 255, B 0 to 139)",
                   "4. Closed: the kept set is closed with a 5 by 5 pixel "
                   "window (dilated, then eroded)",
                   "5. Connected: of that, only the part connected to the "
                   "skeleton (each pixel to its 8 neighbours) is kept.",
                   "6. Every other pixel is set to black, and the picture "
                   "is cut to the bounding box of what is kept.",
                   "This rule, applied to the still, gives the picture "
                   "pixel for pixel.",
                   "The pictures are for the eye only",
                   "grasps\\images\\grasp_images.pdf"):
        assert needle in flat, needle
    assert max(len(x) for x in text.splitlines()) <= 96


def test_two_builds_give_the_same_picture_bytes(sessions, tmp_path):
    grasps = _with_pictures(sessions["grasps"], tmp_path / "a" / "g")
    outs = [tmp_path / "one", tmp_path / "two"]
    for out in outs:
        assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                         "--operator", "N Kim"]) == 0
    names = sorted(p.name for p in (outs[0] / "grasps" / "images").iterdir())
    assert len(names) == 7
    for name in names:
        a, b = (o / "grasps" / "images" / name for o in outs)
        assert a.read_bytes() == b.read_bytes(), name


def test_a_png_the_build_does_not_plan_is_refused(sessions, tmp_path, capsys):
    """Dropped into grasps\\images under a name of its own, a PNG is not one
    of the pictures: the next build refuses to run and writes nothing."""
    grasps = _with_pictures(sessions["grasps"], tmp_path / "a" / "g")
    out = tmp_path / "out"
    args = ["--out", str(out), "--grasps", str(grasps), "--operator", "N Kim"]
    assert pkg.main(args) == 0
    capsys.readouterr()
    images = out / "grasps" / "images"
    (images / "holiday.png").write_bytes(
        (images / f"{STEMS[0]}.png").read_bytes())
    stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert pkg.main(args) == 1
    printed = capsys.readouterr().out
    assert "REFUSED" in printed and "holiday.png" in printed
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*")
            if p.is_file()} == stamps
    # a planned picture's own place is not enough: a copied image is refused
    plan = [pkg.Output(Path("grasps/images/x_left_take1.png"),
                       source=images / f"{STEMS[0]}.png", image_ok=True)]
    assert pkg.allowed_images(plan) == set()
    assert pkg.image_violations(plan)


def test_only_marked_written_pngs_under_images_are_allowed():
    w = lambda dest: None                                     # noqa: E731
    plan = [
        pkg.Output(Path("grasps/images/a_left_take1.png"), write=w,
                   image_ok=True),
        pkg.Output(Path("grasps/images/b_left_take1.png"), write=w),
        pkg.Output(Path("grasps/a_left_take1.png"), write=w, image_ok=True),
        pkg.Output(Path("grasps/images/c_left_take1.jpg"), write=w,
                   image_ok=True),
        pkg.Output(Path("grasps/images/grasp_images.pdf"), write=w)]
    assert pkg.allowed_images(plan) == {"grasps/images/a_left_take1.png"}
    bad = pkg.image_violations(plan)
    assert len(bad) == 3
    assert {Path(b.split(" (")[0]).as_posix() for b in bad} == {
        "grasps/images/b_left_take1.png", "grasps/a_left_take1.png",
        "grasps/images/c_left_take1.jpg"}


def test_a_still_without_the_hand_gives_no_picture_and_says_so(sessions,
                                                                tmp_path):
    grasps = _with_pictures(sessions["grasps"], tmp_path / "a" / "g",
                            without=("tip_pinch_take2",))
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--operator", "N Kim"]) == 0
    rows = {(r["grasp_name"], r["take"]): r for r in _summary(out)}
    assert rows[("tip_pinch", "2")]["image_file"] == ""
    assert rows[("tip_pinch", "1")]["image_file"] == \
        "grasps/images/tip_pinch_left_take1.png"
    assert not (out / "grasps" / "images" / "tip_pinch_left_take2.png").exists()
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert ("1 recording has no picture: tip_pinch_left_take2 (no left "
            "skeleton drawn on its still)") in flat
    assert "per recording whose still shows the hand (5 of 6)" in flat


def test_the_still_notes_become_notes_and_section_1_names_them(
        sessions, tmp_path, monkeypatch):
    """What the pictures show by eye, from the still notes: a note on each
    take it names (so the counts of recordings with a note change with
    them), section 1's list, and section 15's three kinds. Every picture is
    in the notes' "read": nothing is said of a picture not read."""
    gh = pkg.gh
    grasps = _with_pictures(sessions["grasps"], tmp_path / "a" / "g")
    notes = {"session": grasps.name, "read_on": "2026-10-06",
             "confirmed_by_operator": False,
             "in_or_against_the_hand": {"cylindrical_left_take2": "a pen"},
             "in_the_other_hand": {"tip_pinch_left_take1": "a cup"},
             "cannot_tell": ["tip_pinch_left_take3"],
             "read": list(STEMS)}
    monkeypatch.setattr(gh, "load_notes", lambda session: notes)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--operator", "N Kim"]) == 0
    rows = {(r["grasp_name"], r["take"]): r for r in _summary(out)}
    when = "(read by eye on 2026-10-06, not confirmed by the operator)"
    assert rows[("cylindrical", "2")]["notes"].endswith(
        "the infrared picture shows in or against the recorded hand: a pen "
        + when)
    # read off the recorder's uncut still: the other hand is not in the cut
    # picture, and its note says so
    assert rows[("tip_pinch", "1")]["notes"].endswith(
        "the recorder's uncut still shows in the other hand: a cup " + when
        + "; the other hand is not in the cut picture of this folder")
    assert rows[("tip_pinch", "3")]["notes"].endswith(
        "the infrared picture does not settle whether something was in the "
        "hand " + when)
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for needle in (
            "What was seen in the hands, read by eye on 2026-10-06 (not "
            "confirmed by the operator) from the recorder's uncut stills, "
            "which show both hands and the room and are kept in the session "
            "folder, not in this folder; the pictures in this folder "
            "(section 5) are cut to the recorded hand, so what is reported "
            "for the other hand cannot be seen in them:",
            "1 recording with something in or against the recorded hand: "
            "cylindrical_left_take2: a pen",
            "1 with something in the other hand: tip_pinch_left_take1: a cup",
            "1 where the picture does not let one tell: "
            "tip_pinch_left_take3.",
            "the other 3 with a picture: nothing seen in the hand.",
            "Which recordings were made with a real object in the hand is "
            "not known to this folder",
            "What was seen in the hands, read by eye: section 1. The reading "
            "by eye of section 1 was made on the uncut stills, which show "
            "both hands and the room and are kept in the session folder, not "
            "in this folder.",
            "Something in the recorded hand in the picture (section 5): 1",
            "Something in the other hand in the uncut still, not in the cut "
            "picture (section 5): 1",
            "The picture does not settle whether something was in the hand "
            "(section 5): 1",
            "what was in the hand (the pictures are read by eye only"):
        assert needle in flat, needle
    for gone in ("No object was held", "held open", "from the pictures;",
                 "show in the hands", "Another tracked hand",
                 "the infrared picture shows in the other hand"):
        assert gone not in flat, gone
    assert max(len(x) for x in text.splitlines()) <= 96
    assert "—" not in text and "–" not in text
    # the captions of the pictures: "(above)" only where the picture above
    # shows what the caption names; the other hand is not in the picture
    from pypdf import PdfReader
    pages = [" ".join((p.extract_text() or "").split()) for p in PdfReader(
        str(out / "grasps" / "images" / "grasp_images.pdf")).pages]
    page = " ".join(pages)
    assert page.count("by eye, uncut still: something in the other hand "
                      "(not in this picture)") == 1
    assert page.count("by eye, uncut still: something in the hand (above)"
                      ) == 1
    assert page.count("by eye, uncut still: cannot tell") == 1
    assert "cannot tell (above)" not in page
    assert "other hand (above)" not in page and "by eye:" not in page
    assert ("What was seen, read by eye on 2026-10-06 from the recorder's "
            "uncut stills; not confirmed by the operator:") in page
    assert ("take 1: in the other hand, which is not in the picture below: "
            "a cup") in page
    # every picture read: no line, note or kind for a picture not read
    assert "not yet read" not in flat and "not been read" not in flat
    assert "and which have not been read yet" not in flat
    # the same take is not counted twice, and a note is a note
    gs = gh.collect(pc.load_session(grasps), still_notes=notes)
    assert sum(1 for t in gs.takes if t.facts.picture_notes) == 3
    assert all(t.still_read for t in gs.takes) and gs.unread == []
    assert gh.collect(pc.load_session(grasps), still_notes={}).takes[
        1].facts.picture_notes == []


def test_a_picture_not_read_by_eye_is_not_counted_as_showing_nothing(
        sessions, tmp_path, monkeypatch):
    """A take whose picture the notes do not list as read (a take recorded
    after the reading) is on a line of its own in section 1 with a note of
    its own, never among the pictures that showed nothing; the three
    categories keep their counts and the folder passes its check. With
    every picture read, the README is the one a notes file without "read"
    that names every take gives: the list adds nothing then."""
    from pypdf import PdfReader

    from leap_hand import grasp_validate as gv

    gh = pkg.gh
    grasps = _with_pictures(sessions["grasps"], tmp_path / "a" / "g")
    unread = ["cylindrical_left_take3", "tip_pinch_left_take2"]
    notes = {"session": grasps.name, "read_on": "2026-10-06",
             "confirmed_by_operator": False,
             "in_or_against_the_hand": {"cylindrical_left_take2": "a pen"},
             "in_the_other_hand": {"tip_pinch_left_take1": "a cup"},
             "cannot_tell": ["tip_pinch_left_take3"],
             "read": [s for s in STEMS if s not in unread]}
    monkeypatch.setattr(gh, "load_notes", lambda session: notes)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--operator", "N Kim"]) == 0
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for needle in (
            "1 recording with something in or against the recorded hand: "
            "cylindrical_left_take2: a pen",
            "1 with something in the other hand: tip_pinch_left_take1: a cup",
            "1 where the picture does not let one tell: "
            "tip_pinch_left_take3.",
            "the other 1 with a picture: nothing seen in the hand. "
            "2 with a picture not yet read by eye: cylindrical_left_take3, "
            "tip_pinch_left_take2. Which recordings were made with a real "
            "object in the hand is not known to this folder",
            "What was seen in the hands, read by eye, and which pictures "
            "have not been read yet: section 1. The reading by eye of "
            "section 1 was made on the uncut stills,",
            "The picture has not been read by eye (section 5): 2 "
            "cylindrical_left_take3: the infrared picture has not been read "
            "by eye tip_pinch_left_take2: the infrared picture has not been "
            "read by eye"):
        assert needle in flat, needle
    assert max(len(x) for x in text.splitlines()) <= 96
    rows = {f"{r['grasp_name']}_left_take{r['take']}": r
            for r in _summary(out)}
    for stem, r in rows.items():
        assert (r["notes"].endswith("the infrared picture has not been read "
                                    "by eye")) == (stem in unread), stem
    # the picture pages say it too, and never "cannot tell" for it
    pages = [p.extract_text() for p in PdfReader(str(
        out / "grasps" / "images" / "grasp_images.pdf")).pages]
    page = " ".join(" ".join(pages).split())
    assert "take 3: the picture has not been read by eye" in page
    assert page.count("not yet read by eye (above)") == 2
    assert page.count("by eye, uncut still: cannot tell") == 1
    assert "cannot tell (above)" not in page
    # the folder check's count of section 1 (the whole check is run on a
    # recorded session in test_validate_grasp_set.py): the counts add up
    # and the unread are named
    def objects_errors(folder: Path):
        rep = gv.Report(folder=folder)
        _head, summary = gv.read_csv(folder / "grasps" / "grasps_summary.csv")
        gv._check_objects(rep, (folder / "README.txt").read_text(
            encoding="utf-8").splitlines(), summary)
        return rep.errors
    assert objects_errors(out) == []
    gs = gh.collect(pc.load_session(grasps), still_notes=notes)
    assert sorted(t.stem for t in gs.unread) == unread
    assert sorted(t.stem for t in gs.takes if t.still_read) == sorted(
        notes["read"])
    assert gs.problems == []
    # notes from before "read": the takes they name were read, no other
    old = {k: v for k, v in notes.items() if k != "read"}
    gs = gh.collect(pc.load_session(grasps), still_notes=old)
    assert sorted(t.stem for t in gs.takes if t.still_read) == [
        "cylindrical_left_take2", "tip_pinch_left_take1",
        "tip_pinch_left_take3"]
    # a take named but left out of "read" is a problem the build names
    odd = dict(notes, read=[s for s in notes["read"]
                            if s != "tip_pinch_left_take3"])
    gs = gh.collect(pc.load_session(grasps), still_notes=odd)
    assert gs.problems == [
        "tip_pinch_left_take3: the still notes say what its picture shows "
        "but do not list it under \"read\""]
    # every picture read and named: the not-yet-read line, note and kind
    # are gone, and the README is the one of notes without "read" (which
    # then counts the takes it names as read)
    every = dict(notes, read=list(STEMS),
                 cannot_tell=["tip_pinch_left_take3",
                              "cylindrical_left_take1"] + unread)
    named = {k: v for k, v in every.items() if k != "read"}
    texts = []
    for k, these in enumerate((every, named)):
        monkeypatch.setattr(gh, "load_notes", lambda session, n=these: n)
        dest = tmp_path / f"all_read_{k}"
        assert pkg.main(["--out", str(dest), "--grasps", str(grasps),
                         "--operator", "N Kim"]) == 0
        # the two folders' paths differ in one digit only, so the README's
        # lines break in the same places
        texts.append([x.replace(f"all_read_{k}", "all_read_N")
                      for x in (dest / "README.txt").read_text(
                          encoding="utf-8").splitlines()
                      if not x.startswith("Folder built:")])
        assert objects_errors(dest) == []
    flat = " ".join(" ".join(texts[0]).split())
    assert "4 where the picture does not let one tell" in flat
    assert "the other 0 with a picture: nothing seen in the hand." in flat
    assert "not yet read" not in flat and "not been read" not in flat
    assert texts[0] == texts[1]


def test_the_other_hand_is_measured_not_said_to_be_held_open(sessions,
                                                             tmp_path):
    """The fixture has both hands in every frame, in the same pose: the
    summary carries the difference of their finger curl and section 16
    counts it, with no "held open"."""
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(sessions["grasps"]),
                     "--operator", "N Kim"]) == 0
    diffs = [float(r["other_hand_finger_curl_minus_own_deg"])
             for r in _summary(out)]
    assert len(diffs) == 6 and all(abs(d) < 30.0 for d in diffs)
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert ("What shape the other hand was in, measured: in the 6 "
            "recordings with another hand on 50 % of the frames or more"
            ) in flat
    assert ("The two are within 30 degrees of each other in 6; the other "
            "hand is more curled by more than 30 in 0, the labelled hand in "
            "0.") in flat
    assert "The other hand's shape was not read by eye." in flat
    assert "held open" not in flat
    # a second hand model in the file is counted whether it is the other
    # hand or a second fit of the same one (told apart further down)
    assert ("A second tracked hand, or a second fit of the same hand, is in "
            "the file of 6 of the 6 recordings: on 90 % of the frames or "
            "more in 6, on fewer in 0.") in flat
    assert "Another tracked hand" not in flat


# --- glove only: Sets B and C recorded with no camera -----------------------

# What a folder may only say when a camera ran beside the glove.
CAMERA_CLAIMS = ("gloves on + camera", "Ultraleap", "as the reference",
                 "camera_<take>", "abs26", "glove and camera",
                 "trails the camera", "Where the camera followed")


@pytest.fixture(scope="module")
def glove_only(tmp_path_factory):
    root = tmp_path_factory.mktemp("glove_only")
    flexion = fx.make_flexion_session(root / "b", [
        fx.FlexTake("index"),
        fx.FlexTake("ring", coupling={"pinky": 0.4})],
        camera=False, stills=False)
    sequences = fx.make_sequence_session(root / "c", [fx.SeqTake()],
                                         camera=False, stills=False)
    out = root / "glove set"
    rc = pkg.main(["--out", str(out), "--flexion", str(flexion),
                   "--sequences", str(sequences), "--operator", "N Kim"])
    return rc, out, pc.load_session(flexion), pc.load_session(sequences)


def _csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_a_glove_only_folder_says_so_and_claims_no_camera(glove_only):
    rc, out, _flexion, _sequences = glove_only
    assert rc == 0
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for needle in ("Hardware: gloves on, no camera (left hand",
                   "No camera ran: the glove is the only sensor of these "
                   "recordings.",
                   "sensors: glove only, no camera",
                   "No camera ran, so no transfer curve, hysteresis or lag "
                   "is reported.",
                   "the camera columns are empty: no camera ran",
                   "the lag of these recordings is not measured",
                   "no abduction was measured in these recordings",
                   "KNOWN GLOVE LIMITS", "reported as coupling", "60 %"):
        assert needle in flat, needle
    for claim in CAMERA_CLAIMS:
        assert claim not in flat, claim
    assert text.count("sensors: glove only, no camera") == 2
    assert not list(out.rglob("camera_*"))
    report = (out / "finger_flexion" / "left" / "flexion_report.txt"
              ).read_text(encoding="utf-8")
    assert "no camera: the glove is the only sensor" in report
    for claim in CAMERA_CLAIMS + ("camera_trusted", "camera followed",
                                  "nearest camera frame"):
        assert claim not in report, claim
    assert "camera          none in this session (glove only)" in report


def test_the_camera_claims_are_what_a_camera_session_does_say(package):
    """The positive control of the test above: the same words are in the
    folder whose sessions ran the camera, so their absence means something."""
    _rc, out = package
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    report = (out / "finger_flexion" / "left" / "flexion_report.txt"
              ).read_text(encoding="utf-8")
    for claim in CAMERA_CLAIMS:
        assert claim in flat or claim in report, claim
    assert "camera_trusted" in report and "camera followed" in report


def test_a_folder_of_both_kinds_names_the_glove_only_session(package,
                                                             sessions):
    _rc, out = package
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    right = pc.load_session(sessions["flexion_right"])
    assert f"Glove only, no camera: session {right.name}." in flat
    assert flat.count("sensors: glove only, no camera") == 1
    assert flat.count("sensors: glove and Ultraleap camera") == 2
    report = (out / "finger_flexion" / "right" / "flexion_report.txt"
              ).read_text(encoding="utf-8")
    assert "no camera: the glove is the only sensor" in report
    assert "Ultraleap" not in report


def test_every_glove_take_has_its_angles_frame_by_frame(glove_only):
    _rc, out, flexion, sequences = glove_only
    for session, kind in ((flexion, "finger_flexion"),
                          (sequences, "sequences")):
        folder = out / kind / "left" / pkg.ANGLES_DIR
        takes = [t for t in session.takes if t.accepted]
        assert sorted(p.name for p in folder.iterdir()) == sorted(
            f"{t.name}.csv" for t in takes)
        for t in takes:
            rows = _csv(folder / f"{t.name}.csv")
            lines = [json.loads(x) for x in t.path("glove").read_text(
                encoding="utf-8").splitlines() if x.strip()]
            assert list(rows[0]) == pkg.asr.FRAME_COLUMNS
            assert len(rows) == len(lines)
            assert [r["frame_id"] for r in rows] == [
                str(d["frame_id"]) for d in lines]
            cues = [e for e in pc.read_events(t.path("events"))
                    if e["kind"] == "cue"]
            assert {r["cue_step"] for r in rows} - {""} == {
                str(e["step"]) for e in cues}
    # the index take: while the index is cued flexed and held, it is bent
    # and the middle finger is not
    index = next(t for t in flexion.takes if t.item == "index")
    rows = _csv(out / "finger_flexion" / "left" / pkg.ANGLES_DIR
                / f"{index.name}.csv")
    held = [r for r in rows if r["cue_phase"] == "hold"]
    assert held and all(r["flexed"] == "index" for r in held)
    assert max(float(r["I_PIP"]) for r in held) > 30
    assert max(float(r["M_PIP"]) for r in rows) < 5


def test_cue_angles_has_every_held_cue_and_shows_a_finger_going_along(
        glove_only):
    _rc, out, flexion, sequences = glove_only
    rows = _csv(out / "finger_flexion" / "left" / pkg.CUE_ANGLES_FILE)
    assert list(rows[0]) == pkg.asr.CUE_ROW_COLUMNS
    assert len(rows) == 2 * 5 * 2                # takes x cycles x hold, rest
    assert {r["phase"] for r in rows} == {"hold", "rest"}
    assert {r["session"] for r in rows} == {flexion.name}
    ring = [r for r in rows if r["item"] == "ring"]
    # the ring take was built with the little finger following at 0.4
    for hold, rest in zip(ring[0::2], ring[1::2]):
        assert hold["phase"] == "hold" and rest["phase"] == "rest"
        assert hold["cycle"] == rest["cycle"] and int(hold["frames"]) > 10
        assert float(hold["R_PIP"]) > 30 > float(rest["R_PIP"])
        assert float(hold["L_PIP"]) - float(rest["L_PIP"]) > 5
        assert float(hold["R_PIP"]) > float(hold["L_PIP"])
        assert abs(float(hold["I_PIP"]) - float(rest["I_PIP"])) < 2
    seq = _csv(out / "sequences" / "left" / pkg.CUE_ANGLES_FILE)
    assert [r["label"] for r in seq] == [s["label"] for s in fx.SEQ4_STEPS]
    assert [r["flexed"] for r in seq] == [
        " ".join(s["flexed"]) for s in fx.SEQ4_STEPS]
    assert {r["session"] for r in seq} == {sequences.name}
    check = _csv(out / "sequences" / "left" / "sequence_check.csv")
    assert [r["step"] for r in seq] == [r["step"] for r in check]


def test_the_readme_prints_the_span_each_angle_took(glove_only):
    _rc, out, _flexion, _sequences = glove_only
    text = (out / "README.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert pkg.ANGLE_SECTION_TITLE in lines
    files = sorted((out / "finger_flexion" / "left" / pkg.ANGLES_DIR
                    ).iterdir())
    n = sum(len(_csv(p)) for p in files)
    at = lines.index(f"    finger_flexion, left hand, {n} frames")
    block = lines[at + 1:at + 25]
    assert [x.split()[0] for x in block] == pkg.asr.ANGLES
    assert sum(pkg.MODEL_MARK in x for x in block) == 9
    assert {x.split()[0] for x in block if pkg.MODEL_MARK in x} == set(
        pkg.asr.MODEL_ONLY)
    # the printed span is the span of the take files' own column
    values = [float(r["I_PIP"]) for p in files for r in _csv(p)]
    i_pip = next(x for x in block if x.split()[0] == "I_PIP").split()
    assert float(i_pip[1]) == pytest.approx(min(values), abs=0.006)
    assert float(i_pip[3]) == pytest.approx(max(values), abs=0.006)
    flat = " ".join(text.split())
    assert "it streams finger flexion only" in flat
    assert ", ".join(pkg.asr.MODEL_ONLY) in flat


def test_the_readme_describes_spread_frame_id_and_the_clocks_as_they_are(
        package, glove_only):
    """Corrected 2026-10-16 against the Set C recordings: the glove's finger
    spread follows each finger's own bend (its hand model), it is not a
    template constant; frame_id is 0 in XR Trainer's glove lines, it is not
    a frame counter; wall_time is when the recorder wrote a line and
    capture_time when the packet arrived. Both README variants."""
    texts = {}
    for name, (_rc, out, *_rest) in (("camera", package),
                                     ("glove only", glove_only)):
        text = (out / "README.txt").read_text(encoding="utf-8")
        flat = texts[name] = " ".join(text.split())
        assert "template constant" not in flat, name
        assert "own frame counter" not in flat, name
        assert ("Finger spread is not measured by the glove: its hand model "
                "sets each finger's spread from that finger's bend") in flat, name
        assert ("frame_id (copied from the glove line; it is 0 in the glove "
                "lines XR Trainer sends, so it does not number the frames)"
                ) in flat, name
        assert "wall_time (when the recorder wrote the frame" in flat, name
        assert ("wall_time (when the recorder wrote the line, seconds since "
                "1970 on the recording computer)") in flat, name
        assert "when the packet reached the recording computer" in flat, name
        # the glove sets' conventions, the last section, stay on the page
        conventions = text[text.rindex("COORDINATE CONVENTIONS"):]
        assert max(len(x) for x in conventions.splitlines()) <= 96, name
    assert ("capture_time (glove: when the packet reached the recording "
            "computer; camera: when the tracker captured the frame") in texts[
                "camera"]
    assert "Abduction is measured only in the camera files." in texts["camera"]
    # a folder with no camera says nothing about one
    assert ("capture_time (when the packet reached the recording computer, "
            "the same clock)") in texts["glove only"]
    assert "when the tracker captured" not in texts["glove only"]
    assert "bends), so no abduction was measured in these recordings." in texts[
        "glove only"]


def test_the_readme_says_how_far_each_digit_is_one_value(glove_only):
    """The fixture hand bends all of a finger's joints by one angle, so its
    joints are one value exactly, and its thumb never moves."""
    _rc, out, _flexion, _sequences = glove_only
    lines = (out / "README.txt").read_text(encoding="utf-8").splitlines()
    at = lines.index("    finger_flexion, left hand",
                     next(i for i, x in enumerate(lines)
                          if x.strip().startswith("One value per digit.")))
    block = {x.split()[0]: x for x in lines[at + 1:at + 6]}
    assert list(block) == list(pkg.asr.DIGIT_FLEXIONS)
    assert "not judged" in block["thumb"] and "not judged" in block["middle"]
    for digit in ("index", "ring", "little"):
        m = re.search(r"sum moved +([\d.]+) deg; joints within +([\d.]+) deg "
                      r"RMS of one value \(largest: (\w+), (\d+) frames\)",
                      block[digit])
        assert m, block[digit]
        assert float(m.group(1)) > 10 and float(m.group(2)) < 0.5
        assert m.group(3) in pkg.asr.DIGIT_FLEXIONS[digit]


# --- bend speed in the hand-in (the rule of 2026-10-13) -----------------------
def _speed_session(root, rule, hand="left", takes=None):
    from cam_hand import recording_protocol as rp
    meta = {} if rule is None else {"speed_check": rp.speed_rule()}
    return fx.make_flexion_session(
        root, takes or [fx.FlexTake("index")], hand=hand, camera=False,
        stills=False, protocol=fx.SLOW_PROTOCOL, meta=meta)


def test_the_flexion_report_and_readme_carry_the_bend_times_and_the_rule(
        tmp_path):
    session = _speed_session(tmp_path / "s", "ruled")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(session),
                     "--operator", "N Kim"]) == 0
    report = (out / "finger_flexion" / "left" / "flexion_report.txt"
              ).read_text(encoding="utf-8")
    flat_report = " ".join(report.split())
    assert ("bend s and open s = how long the glove's reading took to cross "
            "from 20 % to 80 % of a bend's own rise, and to fall back across "
            "it") in flat_report
    head = next(x for x in report.splitlines() if x.startswith("  take "))
    assert head.split()[:6] == ["take", "finger", "span", "cycles", "bend",
                                "s"]
    row = next(x for x in report.splitlines()
               if x.startswith("  index_left_take1"))
    bend, opened = row.split()[4:6]
    # the fixture hand eases through a 4 s bend: 0.43 of it between the marks
    assert re.fullmatch(r"1\.[67]\d/4", bend), row
    assert re.fullmatch(r"1\.[67]\d/4", opened), row
    assert ("bend-speed rule in force: a bend the cue gives 2 s or more for "
            "must take at least 15 % of the cue's time") in flat_report
    readme = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    for needle in (
            "Set B, bend speed: the protocol asks for a slow bend and a slow "
            "straighten.",
            "is timed from 20 % to 80 % of its own rise",
            "Under the bend-speed rule a bend the cue gives 2 s or more for "
            "must take at least 15 % of the cue's time, or the take was "
            "recorded again; the fast index item is not judged.",
            "Every Set B session of this folder was recorded under it.",
            "The straighten is never judged:",
            "sensors: glove only, no camera bend-speed rule in force:"):
        assert needle in readme, needle
    # a folder of Set B alone, recorded with the glove alone, says nothing
    # about a camera, in the rules of the set it does not hold either
    for claim in CAMERA_CLAIMS:
        assert claim not in readme, claim
        assert claim not in flat_report, claim


def test_the_readme_says_which_sessions_were_under_the_rule(tmp_path):
    ruled = _speed_session(tmp_path / "a", "ruled")
    free = _speed_session(tmp_path / "b", None, hand="right",
                          takes=[fx.FlexTake("index", bend_in=0.3)])
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(ruled), str(free),
                     "--operator", "N Kim"]) == 0
    readme = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert (f"Recorded under it: {ruled.name}; not under it: {free.name} "
            "(their times are reported as they are, not judged).") in readme
    assert readme.count("bend-speed rule in force:") == 1
    assert readme.count("no bend-speed rule in this session") == 1
    right = (out / "finger_flexion" / "right" / "flexion_report.txt"
             ).read_text(encoding="utf-8")
    row = next(x for x in right.splitlines()
               if x.startswith("  index_right_take1"))
    assert re.fullmatch(r"0\.1\d/4", row.split()[4]), row     # the snap
    # handed in all the same: recorded before the rule, it passes as it did
    assert "bend speed not judged (this session has no speed rule)" in right
    assert (out / "finger_flexion" / "right"
            / f"{pc.load_session(free).takes[0].name}.jsonl").is_file()


def test_a_folder_recorded_before_the_rule_says_the_times_were_not_judged(
        glove_only):
    _rc, out, _flexion, _sequences = glove_only
    readme = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert ("No Set B session of this folder was recorded under the "
            "bend-speed rule (it was added on 2026-10-13): the times are "
            "reported as they are, and a cue that gave 4 s for a bend does "
            "not mean the bend took 4 s.") in readme
    report = (out / "finger_flexion" / "left" / "flexion_report.txt"
              ).read_text(encoding="utf-8")
    # the stock fixture's cues are 1 s: timed, shown over the cue
    row = next(x for x in report.splitlines()
               if x.startswith("  index_left_take1"))
    assert re.fullmatch(r"0\.\d\d/1", row.split()[4]), row


def test_a_check_csv_from_before_the_bend_times_is_not_reused(tmp_path,
                                                              capsys):
    session = _speed_session(tmp_path / "s", "ruled")
    csv_path, _txt, _text = pc.write_check(pc.check_session(session))
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [c for c in rows[0] if "bend" not in c and "straighten" not in c
            and c != "speed_rule"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep)
        w.writeheader()
        w.writerows([{k: r[k] for k in keep} for r in rows])
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(session)]) == 0
    assert "check.csv/check.txt are not current" in capsys.readouterr().out
    report = (out / "finger_flexion" / "left" / "flexion_report.txt"
              ).read_text(encoding="utf-8")
    row = next(x for x in report.splitlines()
               if x.startswith("  index_left_take1"))
    assert re.fullmatch(r"1\.[67]\d/4", row.split()[4]), row
    assert "checked while packaging (fixed bands)" in report
