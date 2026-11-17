"""leap_hand.reference_validation and scripts/leap/reference_validation.py.

Every tracker take here is built in the recorder's line format and read back
through replay_core, so the comparison sees exactly what a real session would
give it. The thumb metacarpal's rotation against the palm is built from this
file's own matrices (R_pm = A^T R(fe, aa) Q with explicit rot_y / rot_x,
`truth_rotation`), not from the module's angle code, and the expected numbers
are written out: an exact tracker gives zero error, a reference bias of +3 deg
gives -3, a cross-talk of 0.1 gives a slope of 0.1, a gain of 0.9 a gain error
of -0.1. Reference samples sit exactly on tracker frame times (same_clock) so
no interpolation error enters unless a test asks for one. Groups by name:
spec, format, sync, compat, align, metrics, gaps, split (the gate ledger
GATES_reference_validation.md runs each with -k).
"""
import copy
import csv
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from leap_hand import reference_validation as rv
from leap_hand import replay_core as rc
from leap_hand import tmc_estimate as te
from xr_hand import joint_frames as jf

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs" / "measurement_specification.md"
SIX = ("T_TMC_fe", "T_TMC_aa", "I_CMC", "M_CMC", "R_CMC", "L_CMC")
T0_US, PERIOD_US = 7_000_000_000, 11_111
A_TRUE = jf.rot_x(-20.0) @ jf.rot_y(35.0) @ jf.rot_z(70.0)


def q_true(hand):
    """LeapC's thumb metacarpal frame in the canonical one (x the thumb's flexion
    axis, y dorsal, z toward the base), its long axis 5 deg off -z."""
    s = 1.0 if hand == "right" else -1.0
    return np.array([[0.0, -s, 0.0], [-s, 0.0, 0.0], [0.0, 0.0, -1.0]]) \
        @ jf.rot_y(4.0) @ jf.rot_x(-3.0)


def truth_rotation(fe, aa, hand, order):
    """The canonical rotation, written out here: flexion about e2 (s = +1 right,
    -1 left), abduction about e1, in the stated order."""
    s = 1.0 if hand == "right" else -1.0
    f, a = jf.rot_y(s * fe), jf.rot_x(-aa)
    return f @ a if order == "flexion_then_abduction" else a @ f


def tracker_r_pm(fe, aa, hand, order):
    return A_TRUE.T @ truth_rotation(fe, aa, hand, order) @ q_true(hand)


# --- building a capture -------------------------------------------------------------------
ITEMS = {  # item: (duration s, isolated, truth(tau) -> (fe, aa))
    "tmc_flexion_sweep": (2.0, "flexion_extension",
                          lambda t: (10.0 + 30.0 * math.sin(math.pi * t), 8.0)),
    "tmc_abduction_sweep": (2.0, "abduction_adduction",
                            lambda t: (5.0, 8.0 + 15.0 * (1.0 - math.cos(math.pi * t)))),
    "tmc_hold_flexed": (1.0, None, lambda t: (25.0, 10.0)),
}


def build(tmp, *, hand="left", order="flexion_then_abduction", trials=2,
          tracker=lambda item, trial, fe, aa: (fe, aa), session=None, participant="P01",
          cmc=False, frame_edit=None, every=3):
    """A tracker session and the rows of its reference capture (not yet written).
    Returns dict(folder, takes, rows, hand, session, participant, order).
    `tracker` says what the tracker's thumb encodes; `frame_edit(item, trial, k,
    line)` may change or drop (return None) a frame line."""
    session = session or f"test_{hand}"
    folder = Path(tmp) / "tracker" / session
    folder.mkdir(parents=True)
    t_us, fid, takes, rows = T0_US, 1, [], []
    items = list(ITEMS) + (["cmc_palm_arc"] if cmc else [])
    for trial in range(1, trials + 1):
        for item in items:
            if item == "cmc_palm_arc":
                dur, truth = 1.5, (lambda t: (4.0, 7.0))
            else:
                dur, _iso, truth = ITEMS[item]
            n = int(dur * 1e6 / PERIOD_US)
            lines, times = [], []
            for k in range(n):
                tau = k * PERIOD_US / 1e6
                fe, aa = truth(tau)
                pose = rv.synthetic_pose(hand, 0, 0, order, r_pm=tracker_r_pm(
                    *tracker(item, trial, fe, aa), hand, order))
                line = rv.synthetic_line(hand, 41, fid, t_us + k * PERIOD_US, pose,
                                         t0_us=T0_US)
                fid += 1
                if frame_edit is not None:
                    line = frame_edit(item, trial, k, line)
                if line is None:
                    continue
                lines.append(line)
                times.append(t_us + k * PERIOD_US)
            name = f"{item}_take{trial}_{hand}.jsonl"
            (folder / name).write_text("".join(json.dumps(x) + "\n" for x in lines),
                                       encoding="utf-8")
            (folder / name).with_suffix(rv.META_SUFFIX).write_text(json.dumps(
                {"participant": participant, "session": session, "item": item}),
                encoding="utf-8")
            takes.append({"file": name, "item": item, "take": trial, "frames": len(lines)})
            # reference samples exactly on every `every`-th tracker frame time
            if item == "cmc_palm_arc":
                for k in range(0, n, every):
                    tau = k * PERIOD_US / 1e6
                    rows.append(_row(len(rows), trial, item, (t_us + k * PERIOD_US) / 1e6,
                                     little=12.0 * (1.0 - math.cos(2 * math.pi * tau / dur)) / 2))
            elif ITEMS[item][1] is None:
                rows.append(_row(len(rows), trial, item, (t_us + 0.1e6) / 1e6, *truth(0.0),
                                 t1=(t_us + 0.9e6) / 1e6))
            else:
                for k in range(0, n, every):
                    rows.append(_row(len(rows), trial, item, (t_us + k * PERIOD_US) / 1e6,
                                     *truth(k * PERIOD_US / 1e6)))
            t_us += int((dur + 1.0) * 1e6)
    rv.write_manifest(folder, participant=participant, hand=hand, session=session,
                      tracker_device="test tracker", takes=takes)
    return {"folder": folder, "takes": takes, "rows": rows, "hand": hand,
            "session": session, "participant": participant, "order": order, "cmc": cmc}


def _row(i, trial, movement, t0, fe=None, aa=None, t1=None, little=None):
    return {"sample_id": f"s{i:05d}", "trial": trial, "movement": movement,
            "time_s": f"{t0:.6f}", "time_end_s": "" if t1 is None else f"{t1:.6f}",
            "tracker_file": "", "valid": 1, "exclusion_reason": "",
            "tmc_fe": "" if fe is None else f"{fe:.9f}",
            "tmc_aa": "" if aa is None else f"{aa:.9f}",
            "little_fe": "" if little is None else f"{little:.9f}"}


def header(b, **over):
    h = {
        "format": rv.FORMAT, "data_origin": "measured", "role": "anatomical_reference",
        "participant": {"id": b["participant"]}, "hand": b["hand"], "session": b["session"],
        "device": {"kind": "optical_motion_capture", "make": "TestMake", "model": "T1",
                   "serial": "0001", "software": "test", "sample_rate_hz": 30.0,
                   "derived_from_tracker": False},
        "calibration": {"id": "static-trial-1", "provenance": "test static trial"},
        "clock": {"name": "tracker timestamp clock", "units": "s"},
        "sync": {"method": "same_clock", "uncertainty_s": 0.0},
        "units": {"angle": "deg"},
        "frames": {"thumb_tmc": {
            "proximal_segment": "trapezium", "distal_segment": "first metacarpal",
            "proximal_frame": "test trapezium frame", "distal_frame": "test metacarpal frame",
            "zero": {"id": "isb_neutral", "text": "frames coincident"},
            "rotation_order": b["order"],
            "positive": {"flexion_extension": "ulnar", "abduction_adduction": "palmar"}}},
        "coordinates": [{"column": "tmc_fe", "joint": "thumb_tmc",
                         "movement": "flexion_extension"},
                        {"column": "tmc_aa", "joint": "thumb_tmc",
                         "movement": "abduction_adduction"}],
        "movements": [{"id": k, "joints": ["thumb_tmc"], "isolated": v[1]}
                      for k, v in ITEMS.items()],
    }
    if b["cmc"]:
        h["frames"]["little_cmc"] = {
            "proximal_segment": "carpus", "distal_segment": "fifth metacarpal",
            "proximal_frame": "carpal frame", "distal_frame": "metacarpal frame",
            "zero": {"id": "isb_neutral", "text": "flat palm"},
            "positive": {"flexion_extension": "palmar"}}
        h["coordinates"].append({"column": "little_fe", "joint": "little_cmc",
                                 "movement": "flexion_extension"})
        h["movements"].append({"id": "cmc_palm_arc", "joints": ["little_cmc"],
                               "isolated": "flexion_extension"})
    h.update(over)
    return h


def write_ref(tmp, b, h=None, rows=None):
    columns = ["tmc_fe", "tmc_aa", "little_fe"]
    return rv.write_reference(Path(tmp) / "reference", h or header(b),
                              b["rows"] if rows is None else rows, columns,
                              {"source/export.txt": "device export\n"})


def run_compare(tmp, b, h=None, rows=None, trials=(1,), **kw):
    ref = rv.load_reference(write_ref(tmp, b, h, rows))
    trk = rv.load_tracker_session(b["folder"])
    return rv.compare(ref, trk, trials, **kw)


def mapped_all(report):
    return report["joints"]["thumb_tmc"]["heldout"]["mapped"]["all"]


# --- spec ------------------------------------------------------------------------------------
LABELS = ("Segments and axes", "Reference posture", "Rotation order", "Signs and units",
          "What the tracker supplies", "Additional measurement required",
          "How accuracy is checked")


def spec_sections():
    text = SPEC.read_text(encoding="utf-8")
    heads = list(re.finditer(r"^### (\S+)\s*$", text, flags=re.M))
    out = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():end]
        cut = re.search(r"^## ", body, flags=re.M)
        out[m.group(1)] = body[:cut.start()] if cut else body
    return text, out


def test_spec_names_every_field_for_each_of_the_six_coordinates():
    _text, sections = spec_sections()
    for name in SIX:
        assert name in sections, f"no '### {name}' section"
        for label in LABELS:
            assert f"**{label}:**" in sections[name], f"{name}: no '{label}'"


def test_spec_keeps_paper3_assumptions_apart_from_what_the_dataset_supports():
    text, _ = spec_sections()
    assert "### Paper 3's model assumptions" not in text   # those are ## sections
    for head in ("## Paper 3's model assumptions", "## What this dataset supports",
                 "## Observability, measured on the dataset",
                 "## Device integrations inspected", "## What no calibration can do"):
        assert head in text, head
    for name in ("I_CMC", "M_CMC", "R_CMC", "L_CMC"):
        assert "unobservable" in spec_sections()[1][name]


# --- canonical convention (explicit vectors) ---------------------------------------------
def test_align_canonical_directions_match_hand_written_vectors():
    c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    # right hand: e1 is ulnar, so flexion moves the head toward +e1
    assert np.allclose(rv.tmc_direction(30, 0, "flexion_then_abduction", "right"),
                       (s30, 0.0, c30))
    assert np.allclose(rv.tmc_direction(30, 0, "flexion_then_abduction", "left"),
                       (-s30, 0.0, c30))
    assert np.allclose(rv.tmc_direction(0, 30, "abduction_then_flexion", "left"),
                       (0.0, s30, c30))                     # palmar is +e2
    # the order changes the pair a direction reads as (sin relations written out)
    fe, aa = 30.0, 20.0
    u = rv.tmc_direction(fe, aa, "abduction_then_flexion", "right")
    want_aa = math.degrees(math.asin(math.sin(math.radians(aa)) * math.cos(math.radians(fe))))
    want_fe = math.degrees(math.atan2(math.sin(math.radians(fe)),
                                      math.cos(math.radians(aa)) * math.cos(math.radians(fe))))
    got = rv.tmc_angles(u, "flexion_then_abduction", "right")
    assert got == pytest.approx((want_fe, want_aa), abs=1e-9)
    assert got == pytest.approx((31.5667, 17.2296), abs=1e-3)


def test_align_estimate_axes_match_tmc_estimate():
    """estimate_direction reads tmc_estimate's own (fe, aa) back in its order."""
    for hand in rc.SIDES:
        rest = tracker_r_pm(3.0, 5.0, hand, "flexion_then_abduction")
        cal = te.Calibration(R_rest=rest, session="s", n_transitions=1, n_frames=1,
                             per_transition=[], agreement={}, provenance="test",
                             kind="neutral", hand=hand)
        for fe, aa in ((20.0, 0.0), (0.0, 15.0), (-10.0, 25.0)):
            r_pm = tracker_r_pm(fe, aa, hand, "flexion_then_abduction")
            pos, rot = rv.synthetic_pose(hand, 0, 0, "", r_pm=r_pm)
            est = te.estimate(pos, rot, hand, cal)
            got = rv.tmc_angles(rv.estimate_direction(rest.T @ r_pm, hand),
                                "abduction_then_flexion", hand)
            assert got == pytest.approx((est["T_TMC_fe_est_neutral"],
                                         est["T_TMC_aa_est_neutral"]), abs=1e-9)


# --- align -----------------------------------------------------------------------------------
@pytest.mark.parametrize("hand", rc.SIDES)
@pytest.mark.parametrize("order", rv.ORDERS)
def test_align_recovers_known_rotation_and_long_axis(tmp_path, hand, order):
    b = build(tmp_path, hand=hand, order=order)
    r = run_compare(tmp_path, b)
    mp = r["joints"]["thumb_tmc"]["mapping"]
    assert rc.rotation_angle_deg(np.array(mp["A"]), A_TRUE) < 1e-6
    b_true = q_true(hand).T @ np.array([0.0, 0.0, 1.0])
    assert math.degrees(math.acos(min(1.0, float(np.dot(mp["b"], b_true))))) < 1e-5
    assert mp["b_from_tracker_axis_deg"] == pytest.approx(
        math.degrees(math.acos(float(b_true @ rv.HEAD))), abs=1e-5)
    assert mp["rms_residual_deg"] < 1e-6
    al = mapped_all(r)
    for c in ("fe", "aa"):
        assert al[c]["mae_deg"] < 1e-6 and al[c]["n"] > 100


def test_align_refuses_a_single_posture_or_a_single_plane():
    r_pm = [tracker_r_pm(10.0, 8.0, "left", "flexion_then_abduction")] * 20
    d = [rv.tmc_direction(10.0, 8.0, "flexion_then_abduction", "left")] * 20
    with pytest.raises(rv.ReferenceError, match="do not determine the mapping"):
        rv.fit_mapping(r_pm, d)
    sweep = np.linspace(-20, 40, 30)
    r_pm = [tracker_r_pm(f, 8.0, "left", "flexion_then_abduction") for f in sweep]
    d = [rv.tmc_direction(f, 8.0, "flexion_then_abduction", "left") for f in sweep]
    with pytest.raises(rv.ReferenceError, match="one posture or one movement plane"):
        rv.fit_mapping(r_pm, d)


def test_align_a_wrongly_declared_order_is_nearly_invisible_in_the_errors(tmp_path):
    """The comparison reads the reference and the mapped values in the same
    declared order, so a wrong declaration leaves only the small non-rigid
    part of the order change in the errors (0.12 deg here): the order cannot
    be told from the data and must come from the device's definition, which
    is why the importer refuses a file that does not declare it."""
    b = build(tmp_path, order="abduction_then_flexion")
    right = mapped_all(run_compare(tmp_path, b))
    h = header(b)
    h["frames"]["thumb_tmc"]["rotation_order"] = "flexion_then_abduction"
    wrong = mapped_all(run_compare(tmp_path / "w", b, h))
    assert max(right["fe"]["abs_max_deg"], right["aa"]["abs_max_deg"]) < 1e-6
    assert 1e-3 < max(wrong["fe"]["abs_max_deg"], wrong["aa"]["abs_max_deg"]) < 1.0


# --- format ----------------------------------------------------------------------------------
def test_format_a_valid_capture_imports_with_its_conventions(tmp_path):
    b = build(tmp_path, trials=1)
    rows = copy.deepcopy(b["rows"])
    for r in rows:                         # the file: flexion positive radial, in radians
        for k in ("tmc_fe", "tmc_aa"):
            if r[k]:
                v = math.radians(float(r[k]))
                r[k] = f"{-v if k == 'tmc_fe' else v:.12f}"
    h = header(b, units={"angle": "rad"})
    h["frames"]["thumb_tmc"]["positive"]["flexion_extension"] = "radial"
    ref = rv.load_reference(write_ref(tmp_path, b, h, rows))
    assert ref.status == "evaluated" and ref.angle_unit == "rad"
    first = next(s for s in ref.samples if s.movement == "tmc_flexion_sweep")
    assert first.values[("thumb_tmc", "flexion_extension")] == pytest.approx(10.0, abs=1e-8)
    assert ref.coordinates[0].sign == -1.0 and ref.coordinates[1].sign == 1.0


def _edit(path, fn):
    d = json.loads(path.read_text(encoding="utf-8"))
    fn(d)
    path.write_text(json.dumps(d), encoding="utf-8")


REFUSALS = [
    ("units", lambda d: d.pop("units"), "units.angle is missing"),
    ("unit", lambda d: d.update(units={"angle": "grad"}), "units.angle must be one of"),
    ("sign", lambda d: d["frames"]["thumb_tmc"]["positive"].update(
        flexion_extension="toward the palm"), "an ambiguous sign is not guessed"),
    ("order", lambda d: d["frames"]["thumb_tmc"].pop("rotation_order"),
     "rotation_order must be one of"),
    ("joint", lambda d: d["coordinates"][0].update(joint="thumb_cmc"), "joint must be one of"),
    ("movement", lambda d: d["coordinates"][0].update(movement="twist"),
     "movement of thumb_tmc must be one of"),
    ("hand", lambda d: d.update(hand="both"), "hand must be one of"),
    ("derived", lambda d: d["device"].update(derived_from_tracker=True),
     "software cross-check, not an independent reference"),
    ("glove", lambda d: d["device"].update(kind="instrumented_glove"),
     "role 'independent_sensor'"),
    ("participant", lambda d: d.pop("participant"), "participant: 'id' is missing"),
    ("one_angle", lambda d: d["coordinates"].pop(1), "needs both its flexion_extension"),
    ("calibration", lambda d: d.pop("calibration"), "'calibration' is missing"),
    ("cal_hand", lambda d: d["calibration"].update(hand="right"),
     "calibration is of the right hand"),
    ("uncertainty", lambda d: d["sync"].pop("uncertainty_s"), "sync.uncertainty_s must be"),
    ("clock", lambda d: d.pop("clock"), "clock.units must be one of"),
    ("events", lambda d: d.update(sync={"method": "shared_events", "uncertainty_s": 0.01}),
     "needs sync.events"),
    ("event_both", lambda d: d.update(sync={"method": "shared_events", "uncertainty_s": 0.01,
                                             "events": [{"id": "e", "reference_time_s": 1.0,
                                                         "tracker_timestamp_us": 5,
                                                         "tracker_file": "x", "tap_index": 1}]}),
     "not both and not neither"),
    ("zero", lambda d: d["frames"]["thumb_tmc"].update(zero={"id": "rest"}), "zero.id must be"),
    ("frames", lambda d: d.pop("frames"), "'frames' is missing"),
    ("movements", lambda d: d.update(movements=[]), "'movements' is missing"),
    ("format", lambda d: d.update(format="v0"), "format must be"),
]


@pytest.mark.parametrize("name,edit,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_format_refuses_with_an_actionable_message(tmp_path, name, edit, message):
    b = build(tmp_path, trials=1)
    folder = write_ref(tmp_path, b)
    _edit(folder / rv.HEADER_FILE, edit)
    with pytest.raises(rv.ReferenceError, match=re.escape(message)):
        rv.load_reference(folder)


def test_format_refuses_edited_or_broken_samples_and_sources(tmp_path):
    b = build(tmp_path, trials=1)
    folder = write_ref(tmp_path, b)
    text = (folder / "samples.csv").read_text(encoding="utf-8")
    (folder / "samples.csv").write_text(text.replace("10.000000000", "11.000000000", 1),
                                        encoding="utf-8")
    with pytest.raises(rv.ReferenceError, match="edited since the header was written"):
        rv.load_reference(folder)
    (folder / "samples.csv").write_text(text, encoding="utf-8")
    (folder / "source" / "export.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(rv.ReferenceError, match="does not match its sha256"):
        rv.load_reference(folder)
    for rows_edit, message in (
            (lambda rs: [r.pop("tmc_aa") for r in rs], "columns missing: tmc_aa"),
            (lambda rs: rs[0].update(valid=0), "needs its exclusion_reason"),
            (lambda rs: rs[0].update(movement="tmc_twist"), "is not in the header's movements"),
            (lambda rs: rs[0].update(time_s=""), "time_s is empty"),
            (lambda rs: rs[-1].update(time_end_s="0.5"), "time_end_s is before time_s"),
            (lambda rs: rs[1].update(sample_id=rs[0]["sample_id"]), "empty or repeated"),
            (lambda rs: rs[0].update(tmc_fe="nan"), "not a finite number")):
        rows = copy.deepcopy(b["rows"])
        rows_edit(rows)
        f2 = tmp_path / f"r{abs(hash(message))}"
        cols = [c for c in ("tmc_fe", "tmc_aa", "little_fe") if c in rows[0]]
        rv.write_reference(f2, header(b), rows, cols, {"source/export.txt": "x\n"})
        with pytest.raises(rv.ReferenceError, match=re.escape(message)):
            rv.load_reference(f2)


def test_format_a_software_cross_check_is_never_evaluated(tmp_path):
    b = build(tmp_path, trials=2)
    h = header(b, data_origin="software_cross_check", role="software_cross_check")
    h["device"]["derived_from_tracker"] = True
    r = run_compare(tmp_path, b, h)
    assert r["status"] == "cross_check" and r["_mapping"].status == "cross_check"
    ref = rv.load_reference(write_ref(tmp_path / "syn", b, header(b, data_origin="synthetic")))
    assert ref.status == "synthetic_demonstration"


def test_format_cli_check_and_compare(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "rv_cli", REPO / "scripts" / "leap" / "reference_validation.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    b = build(tmp_path)
    folder = write_ref(tmp_path, b)
    assert cli.main(["check", str(folder)]) == 0
    assert "REFERENCE OK: participant=P01 hand=left" in capsys.readouterr().out
    out = tmp_path / "cmp"
    assert cli.main(["compare", "--reference", str(folder), "--tracker-session",
                     str(b["folder"]), "--calibration-trials", "1", "--out", str(out)]) == 0
    line = capsys.readouterr().out
    assert "COMPARISON DONE: data_origin=measured status=evaluated" in line
    for name in ("report.json", "report.md", "matched_samples.csv", "mapping.json"):
        assert (out / name).is_file()
    _edit(folder / rv.HEADER_FILE, lambda d: d.update(hand="right"))
    assert cli.main(["check", str(folder)]) == 0     # a right-hand capture on its own is fine
    assert cli.main(["compare", "--reference", str(folder), "--tracker-session",
                     str(b["folder"]), "--calibration-trials", "1", "--out", str(out)]) == 2
    assert "COMPARISON REFUSED: the reference capture is of hand 'right'" in capsys.readouterr().out


# --- sync ------------------------------------------------------------------------------------
def _shared_events(b, offset_s, scale, jitter=(0.0, 0.0, 0.0)):
    """Reference times = (tracker seconds - offset) / scale; three events given
    as tracker_timestamp_us at the first frame of three takes."""
    rows = copy.deepcopy(b["rows"])
    for r in rows:
        for k in ("time_s", "time_end_s"):
            if r[k]:
                r[k] = f"{(float(r[k]) - offset_s) / scale:.9f}"
    tk = [b["takes"][0], b["takes"][2], b["takes"][-1]]
    events = []
    for i, t in enumerate(tk):
        first = json.loads((b["folder"] / t["file"]).read_text(encoding="utf-8").splitlines()[0])
        us = first["timestamp_us"]
        events.append({"id": f"e{i}", "tracker_timestamp_us": us,
                       "reference_time_s": (us / 1e6 - offset_s) / scale + jitter[i]})
    return rows, events


def test_sync_recovers_offset_and_drift_from_paired_events(tmp_path):
    b = build(tmp_path)
    rows, events = _shared_events(b, 6995.5, 1.0 + 50e-6)
    h = header(b, sync={"method": "shared_events", "uncertainty_s": 0.002, "events": events})
    r = run_compare(tmp_path, b, h, rows)
    assert r["sync"]["offset_s"] == pytest.approx(6995.5, abs=1e-6)
    assert r["sync"]["drift"] == pytest.approx(50e-6, abs=1e-9)
    assert r["sync"]["residual_max_s"] < 1e-6
    assert mapped_all(r)["fe"]["mae_deg"] < 1e-3          # times written to 1 ns


@pytest.mark.parametrize("units,scale", [("ms", 1e3), ("us", 1e6)])
def test_sync_same_clock_in_other_time_units(tmp_path, units, scale):
    """The tracker's own clock written in ms or us: converted once, exact."""
    b = build(tmp_path)
    rows = copy.deepcopy(b["rows"])
    for r in rows:
        for k in ("time_s", "time_end_s"):
            if r[k]:
                r[k] = f"{float(r[k]) * scale:.3f}"
    h = header(b, clock={"name": "tracker timestamp clock", "units": units})
    r = run_compare(tmp_path, b, h, rows)
    assert r["sync"]["scale"] == 1.0 and r["sync"]["offset_s"] == 0.0
    al = mapped_all(r)
    assert al["fe"]["n"] > 100 and al["fe"]["mae_deg"] < 1e-3
    assert r["samples"]["excluded"] == 0


def test_sync_one_event_gives_the_offset_only(tmp_path):
    b = build(tmp_path)
    rows, events = _shared_events(b, 6995.5, 1.0)
    h = header(b, sync={"method": "hardware_trigger", "uncertainty_s": 0.001,
                        "events": events[:1]})
    r = run_compare(tmp_path, b, h, rows)
    assert r["sync"]["drift"] == 0.0 and "drift not measured" in r["sync"]["note"]


def test_sync_refuses_events_that_disagree_or_imply_a_wild_drift(tmp_path):
    b = build(tmp_path)
    rows, events = _shared_events(b, 6995.5, 1.0, jitter=(0.0, 0.03, 0.0))
    h = header(b, sync={"method": "shared_events", "uncertainty_s": 0.01, "events": events})
    with pytest.raises(rv.ReferenceError, match="more than the declared uncertainty 10.0 ms"):
        run_compare(tmp_path, b, h, rows)
    rows, events = _shared_events(b, 6995.5, 1.05)
    h = header(b, sync={"method": "shared_events", "uncertainty_s": 0.01, "events": events})
    with pytest.raises(rv.ReferenceError, match="probably mispaired"):
        run_compare(tmp_path / "w", b, h, rows)


def _tap_take(tmp, hand="left", taps=(1.0, 2.0, 3.0), n=400, cut_after=None):
    folder = Path(tmp)
    lines = []
    for k in range(n):
        tau = k * PERIOD_US / 1e6
        gap = 60.0 - 55.0 * max(max(0.0, 1.0 - abs(tau - c) / 0.25) for c in taps)
        pose = rv.synthetic_pose(hand, 5.0, 7.0, "flexion_then_abduction", tip_gap_mm=gap)
        line = rv.synthetic_line(hand, 41, 500 + k, T0_US + k * PERIOD_US, pose, t0_us=T0_US)
        if cut_after is not None and tau > cut_after:
            line["hand_id"] = 42                           # an identity change mid-tap
        lines.append(line)
    (folder / "sync_start_take1_left.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return rc.load_recording(folder / "sync_start_take1_left.jsonl", hand)


def test_sync_taps_are_detected_at_the_closest_frame(tmp_path):
    rec = _tap_take(tmp_path)
    taps = rv.tap_times(rec)
    assert len(taps) == 3
    for tap, c in zip(taps, (1.0, 2.0, 3.0)):
        k = round(c * 1e6 / PERIOD_US)
        assert tap["timestamp_us"] == T0_US + k * PERIOD_US
        assert tap["distance_mm"] < 10.0
    # a tap cut by an identity change is dropped, not half-counted
    (tmp_path / "cut").mkdir()
    rec2 = _tap_take(tmp_path / "cut", cut_after=2.0)
    assert len(rv.tap_times(rec2)) == 2


def test_sync_events_resolve_from_detected_taps_and_a_missing_tap_is_refused(tmp_path):
    b = build(tmp_path)
    rec = _tap_take(b["folder"])
    taps = rv.tap_times(rec)
    takes = b["takes"] + [{"file": "sync_start_take1_left.jsonl", "item": "sync_start",
                           "take": 1, "frames": len(rec.frames)}]
    rv.write_manifest(b["folder"], participant="P01", hand="left", session=b["session"],
                      tracker_device="test tracker", takes=takes)
    offset = 6990.0
    rows = copy.deepcopy(b["rows"])
    for r in rows:
        for k in ("time_s", "time_end_s"):
            if r[k]:
                r[k] = f"{float(r[k]) - offset:.9f}"
    events = [{"id": f"tap{k}", "tracker_file": "sync_start_take1_left.jsonl", "tap_index": k,
               "reference_time_s": taps[k - 1]["timestamp_us"] / 1e6 - offset}
              for k in (1, 2, 3)]
    h = header(b, sync={"method": "shared_events", "uncertainty_s": 0.001, "events": events})
    r = run_compare(tmp_path, b, h, rows)
    assert r["sync"]["offset_s"] == pytest.approx(offset, abs=1e-6)
    assert all("tap" in e["resolved"] for e in r["sync"]["events"])
    events.append({"id": "tap4", "tracker_file": "sync_start_take1_left.jsonl",
                   "tap_index": 4, "reference_time_s": 5.0})
    h = header(b, sync={"method": "shared_events", "uncertainty_s": 0.001, "events": events})
    with pytest.raises(rv.ReferenceError, match="but 3 taps were detected there"):
        run_compare(tmp_path / "m", b, h, rows)


def test_sync_sensitivity_moves_the_errors_by_the_declared_uncertainty(tmp_path):
    b = build(tmp_path, every=1)
    h = header(b, sync={"method": "same_clock", "uncertainty_s": 0.02})
    r = run_compare(tmp_path, b, h)
    s = r["joints"]["thumb_tmc"]["sync_sensitivity"]
    assert s["zero"]["fe_rmse_deg"] < 1e-6
    assert s["minus"]["shift_s"] == -0.02 and s["plus"]["shift_s"] == 0.02
    # a 30 deg sine at 0.5 Hz moves up to 30 pi 0.5 deg/s: 20 ms is up to ~0.94 deg
    assert 0.1 < s["plus"]["fe_rmse_deg"] < 1.0 and 0.1 < s["minus"]["fe_rmse_deg"] < 1.0


# --- compat ----------------------------------------------------------------------------------
def test_compat_reference_and_tracker_session_must_be_one_capture(tmp_path):
    b = build(tmp_path)
    for over, word in (({"hand": "right"}, "hand"), ({"participant": {"id": "P02"}}, "participant"),
                       ({"session": "other"}, "session")):
        with pytest.raises(rv.ReferenceError, match=f"the reference capture is of {word}"):
            run_compare(tmp_path / word, b, header(b, **over))


def test_compat_tracker_calibration_must_be_of_the_hand_and_the_named_file(tmp_path):
    b = build(tmp_path)
    rest = tracker_r_pm(4.0, 7.0, "left", b["order"])

    def cal(hand):
        return te.Calibration(R_rest=rest, session=b["session"], n_transitions=1, n_frames=1,
                              per_transition=[], agreement={}, provenance="test",
                              kind="neutral", hand=hand)
    with pytest.raises(rv.ReferenceError, match="calibration of the right hand"):
        run_compare(tmp_path, b, tmc_calibration=cal("right"))
    with pytest.raises(rv.ReferenceError, match="calibration hand not recorded"):
        run_compare(tmp_path / "u", b, tmc_calibration=cal(None))
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(cal("left").to_json()), encoding="utf-8")
    h = header(b, tracker_calibration={"file": "other.json", "sha256": "0" * 64})
    with pytest.raises(rv.ReferenceError, match="names the tracker calibration"):
        run_compare(tmp_path / "s", b, h, tmc_calibration=cal("left"),
                    tmc_calibration_path=path)
    h = header(b, tracker_calibration={"file": "cal.json", "sha256": rv.sha256_of(path)})
    r = run_compare(tmp_path / "ok", b, h, tmc_calibration=cal("left"),
                    tmc_calibration_path=path)
    est = r["joints"]["thumb_tmc"]["heldout"]["estimate_neutral"]["all"]
    assert est["fe"]["n"] == mapped_all(r)["fe"]["n"]
    assert abs(est["fe"]["bias_deg"]) > 1.0       # a posture zero is not the reference's zero


def test_compat_a_mapping_applies_only_inside_its_scope(tmp_path):
    b = build(tmp_path)
    r = run_compare(tmp_path, b)
    rv.write_report(r, tmp_path / "out")
    m = rv.load_mapping(tmp_path / "out" / "mapping.json")
    assert m.status == "evaluated"
    ok = rv.scope_mismatch(m, "left", b["session"], "P01")
    assert ok is None
    assert rv.scope_mismatch(m, "right", b["session"], "P01") == (
        "this recording is of the right hand, the mapping of the left hand")
    assert rv.scope_mismatch(m, "left", "other", "P01") == (
        "this recording is of session other, the mapping of session test_left")
    assert rv.scope_mismatch(m, "left", b["session"], "P02") == (
        "this recording is of participant P02, the mapping of participant P01")
    assert rv.scope_mismatch(m, "left", b["session"], None).startswith(
        "this recording names no participant (the mapping is of participant P01")
    rec = rc.load_recording(b["folder"] / b["takes"][0]["file"], "left")
    assert rv.recording_scope(rec) == {"session": b["session"], "participant": "P01"}
    # inside the tested range a value, outside it none
    pos, rot = rv.synthetic_pose("left", 0, 0, "", r_pm=tracker_r_pm(10.0, 8.0, "left",
                                                                     b["order"]))
    v = rv.apply_mapping(m, rot)
    assert v["status"] == "evaluated_in_range"
    assert (v["T_TMC_fe_mapped"], v["T_TMC_aa_mapped"]) == pytest.approx((10.0, 8.0), abs=1e-6)
    pos, rot = rv.synthetic_pose("left", 0, 0, "", r_pm=tracker_r_pm(70.0, 8.0, "left",
                                                                     b["order"]))
    v = rv.apply_mapping(m, rot)
    assert v["status"] == "outside_tested_range" and v["T_TMC_fe_mapped"] is None
    d = json.loads((tmp_path / "out" / "mapping.json").read_text(encoding="utf-8"))
    d["A"][0][0] = 2.0
    (tmp_path / "bad.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(rv.ReferenceError, match="must be a 3x3 rotation"):
        rv.load_mapping(tmp_path / "bad.json")


# --- metrics ---------------------------------------------------------------------------------
def test_metrics_error_statistics_written_out():
    s = rv.error_stats([1.0, -1.0, 3.0, 5.0], [0.0, 10.0, 20.0, 30.0])
    assert s["n"] == 4 and s["bias_deg"] == 2.0 and s["mae_deg"] == 2.5
    assert s["rmse_deg"] == pytest.approx(math.sqrt((1 + 1 + 9 + 25) / 4))
    assert s["abs_p50_deg"] == 2.0 and s["abs_max_deg"] == 5.0
    assert s["abs_p90_deg"] == pytest.approx(4.4)            # numpy linear percentile
    assert (s["reference_min_deg"], s["reference_max_deg"]) == (0.0, 30.0)


def test_metrics_wrap_angle_differences_across_180():
    assert float(rv.wrap180(179.0 - (-179.0))) == pytest.approx(-2.0)
    assert float(rv.wrap180(-179.0 - 179.0)) == pytest.approx(2.0)
    s = rv.error_stats([358.0, -358.0], [0.0, 0.0])
    assert s["mae_deg"] == pytest.approx(2.0) and s["bias_deg"] == pytest.approx(0.0)


def test_metrics_crosstalk_gain_and_repeatability_written_out():
    rows = []
    for trial, bias in ((1, 1.0), (2, 2.0), (3, 3.0)):
        for fe in np.linspace(-10, 30, 9):
            rows.append({"trial": trial, "movement": "sweep",
                         "ref": {"fe": fe, "aa": 8.0},
                         "pred": {"fe": 0.9 * fe + bias, "aa": 8.0 + 0.1 * fe}})
    ev = rv.evaluate(rows, {"sweep": "flexion_extension"}, ("fe", "aa"))
    ct = ev["sweep"]["crosstalk"]
    assert ct["gain_error_slope_deg_per_deg"] == pytest.approx(-0.1)
    assert ct["into"]["aa"]["error_slope_deg_per_deg"] == pytest.approx(0.1)
    assert ct["into"]["aa"]["error_range_deg"] == pytest.approx(4.0)
    assert ct["into"]["aa"]["reference_offaxis_slope"] == pytest.approx(0.0)
    rep = ev["sweep"]["repeatability"]["fe"]
    # each trial's mean error: bias - 0.1 * mean(fe) = bias - 1.0
    assert rep["per_trial_bias_deg"] == pytest.approx({1: 0.0, 2: 1.0, 3: 2.0})
    assert rep["sd_of_trial_bias_deg"] == pytest.approx(1.0)
    assert rep["range_of_trial_bias_deg"] == pytest.approx(2.0)
    one = rv.evaluate(rows[:9], {"sweep": "flexion_extension"}, ("fe", "aa"))
    assert one["sweep"]["repeatability"]["fe"]["sd_of_trial_bias_deg"] is None
    still = [{**r, "ref": {"fe": 5.0, "aa": 8.0}} for r in rows[:9]]
    assert rv.evaluate(still, {"sweep": "flexion_extension"},
                       ("fe", "aa"))["sweep"]["crosstalk"]["gain_error_slope_deg_per_deg"] is None


def test_metrics_end_to_end_bias_gain_and_crosstalk(tmp_path):
    """Calibration trial 1 exact; held-out trial 2 with a known defect."""
    def xtalk(item, trial, fe, aa):
        return (fe, aa + 0.1 * fe) if trial == 2 else (fe, aa)
    r = run_compare(tmp_path, build(tmp_path, tracker=xtalk))
    sweep = r["joints"]["thumb_tmc"]["heldout"]["mapped"]["tmc_flexion_sweep"]
    assert sweep["crosstalk"]["into"]["aa"]["error_slope_deg_per_deg"] == pytest.approx(0.1, abs=1e-6)
    assert sweep["crosstalk"]["gain_error_slope_deg_per_deg"] == pytest.approx(0.0, abs=1e-6)

    def gain(item, trial, fe, aa):
        return (0.9 * fe, aa) if trial == 2 else (fe, aa)
    r = run_compare(tmp_path / "g", build(tmp_path / "g", tracker=gain))
    sweep = r["joints"]["thumb_tmc"]["heldout"]["mapped"]["tmc_flexion_sweep"]
    assert sweep["crosstalk"]["gain_error_slope_deg_per_deg"] == pytest.approx(-0.1, abs=1e-6)

    b = build(tmp_path / "b")
    rows = copy.deepcopy(b["rows"])
    for row in rows:                                   # the reference reads +3 in trial 2
        if row["trial"] == 2 and row["tmc_fe"]:
            row["tmc_fe"] = f"{float(row['tmc_fe']) + 3.0:.9f}"
    r = run_compare(tmp_path / "b", b, rows=rows)
    al = mapped_all(r)
    assert al["fe"]["bias_deg"] == pytest.approx(-3.0, abs=1e-6)
    assert al["fe"]["mae_deg"] == pytest.approx(3.0, abs=1e-6)
    assert al["fe"]["rmse_deg"] == pytest.approx(3.0, abs=1e-6)
    assert al["aa"]["mae_deg"] < 1e-6
    # the tested range is the held-out reference's (here carrying the +3 deg),
    # the calibration range is reported apart: fitted, not tested
    tested = r["joints"]["thumb_tmc"]["tested_range_deg"]
    fitted = r["joints"]["thumb_tmc"]["calibration_range_deg"]
    assert tested["fe"] == pytest.approx([-17.0, 43.0], abs=0.05)
    assert fitted["fe"] == pytest.approx([-20.0, 40.0], abs=0.05)
    assert tested["aa"][1] == pytest.approx(38.0, abs=0.05)


# --- gaps ------------------------------------------------------------------------------------
def test_gaps_every_exclusion_names_its_reason_and_is_counted(tmp_path):
    def edit(item, trial, k, line):
        if (item, trial) == ("tmc_flexion_sweep", 2) and 60 <= k < 80:
            return None                                          # a 0.22 s gap
        if (item, trial) == ("tmc_abduction_sweep", 2) and k >= 90:
            line["hand_id"] = 42                                 # an identity change
        if (item, trial) == ("tmc_abduction_sweep", 2) and k == 40:
            line["status"] = 0                                   # invalid tracking
        return line
    b = build(tmp_path, frame_edit=edit, every=1000)            # one instant per sweep
    rows, n = [], 0
    takes = {(t["item"], t["take"]): t for t in b["takes"]}
    first_us = {}
    for t in b["takes"]:
        lines = (b["folder"] / t["file"]).read_text(encoding="utf-8").splitlines()
        first_us[(t["item"], t["take"])] = json.loads(lines[0])["timestamp_us"]

    def at(item, trial, k_float, **kw):
        nonlocal n
        n += 1
        t = (first_us[(item, trial)] + k_float * PERIOD_US) / 1e6
        rows.append({**_row(n, trial, item, t, 10.0, 8.0), **kw})
    for item in ITEMS:                              # trial 1: clean calibration data
        for k in range(0, 170, 3):
            at(item, 1, k) if ITEMS[item][1] else None
    rows += [{**r, "sample_id": "c" + r["sample_id"]} for r in b["rows"]
             if r["trial"] == 1 and r["time_end_s"]]
    for k in range(0, 170, 3):                      # trial 2: clean held-out samples
        if not (55 <= k <= 85):
            at("tmc_abduction_sweep", 2, k) if not (35 <= k <= 95) else None
    at("tmc_flexion_sweep", 2, 70.5)                                  # in the gap
    at("tmc_abduction_sweep", 2, 89.5)                                # across the id change
    at("tmc_abduction_sweep", 2, 39.5)                                # across the status-0 line
    at("tmc_abduction_sweep", 2, 10, valid=0, exclusion_reason="marker occluded")
    at("tmc_abduction_sweep", 2, 12, tmc_aa="")                       # a missing coordinate
    at("tmc_abduction_sweep", 2, 200)                                 # after the take
    at("tmc_abduction_sweep", 2, 14, movement="tmc_flexion_sweep")    # another movement's take
    at("tmc_abduction_sweep", 2, 16, movement="")                     # unlabelled
    hold = next(r for r in b["rows"] if r["trial"] == 2 and r["time_end_s"])
    rows.append({**hold, "sample_id": "hold_ok"})
    t0 = first_us[("tmc_flexion_sweep", 2)] / 1e6
    rows.append(_row(9999, 2, "tmc_flexion_sweep", t0 + 0.4, 10.0, 8.0, t1=t0 + 1.2))
    r = run_compare(tmp_path, b, rows=rows)
    by_id = {m.sample.sample_id: m for m in r["_matches"]}
    want = {f"s{n - 7:05d}": "tracker_discontinuity", f"s{n - 6:05d}": "tracker_discontinuity",
            f"s{n - 5:05d}": "tracker_discontinuity", f"s{n - 4:05d}": "reference_invalid",
            f"s{n - 3:05d}": "reference_coordinate_missing",
            f"s{n - 2:05d}": "outside_tracker_takes", f"s{n - 1:05d}": "movement_mismatch",
            f"s{n:05d}": "reference_unlabeled", "s09999": "tracker_discontinuity inside the hold"}
    for sid, reason in want.items():
        assert by_id[sid].reason.startswith(reason), (sid, by_id[sid].reason)
        assert not by_id[sid].used
    assert "gap" in by_id[f"s{n - 7:05d}"].reason
    assert "id_change" in by_id[f"s{n - 6:05d}"].reason
    assert "invalid_status" in by_id[f"s{n - 5:05d}"].reason
    assert "marker occluded" in by_id[f"s{n - 4:05d}"].reason
    assert "tmc_aa" in by_id[f"s{n - 3:05d}"].reason
    assert by_id["hold_ok"].used and by_id["hold_ok"].n_frames >= rv.MIN_HOLD_FRAMES
    counts = r["samples"]["excluded_by_reason"]
    assert counts["tracker_discontinuity"] == 3
    assert counts["tracker_discontinuity inside the hold"] == 1
    for k in ("reference_invalid", "reference_coordinate_missing", "outside_tracker_takes",
              "movement_mismatch", "reference_unlabeled"):
        assert counts[k] == 1, k
    assert r["samples"]["excluded"] == sum(counts.values()) == 9
    rv.write_report(r, tmp_path / "w")
    with open(tmp_path / "w" / "matched_samples.csv", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert sum(row["used"] == "0" for row in csv_rows) == 9
    assert all(row["reason"] for row in csv_rows if row["used"] == "0")


def test_gaps_a_take_off_the_camera_clock_is_excluded_not_guessed(tmp_path):
    def edit(item, trial, k, line):
        if (item, trial) == ("tmc_flexion_sweep", 2) and k == 50:
            line["timestamp_us"] -= 2 * PERIOD_US          # the clock goes backwards
        return line
    b = build(tmp_path, frame_edit=edit)
    r = run_compare(tmp_path, b)
    reasons = {m.reason.split(":")[0] for m in r["_matches"]
               if m.sample.trial == 2 and m.sample.movement == "tmc_flexion_sweep"}
    assert reasons == {"tracker_clock_fallback"}


def test_gaps_the_finger_cmc_is_unobservable_and_nothing_is_fitted(tmp_path):
    b = build(tmp_path, cmc=True)
    r = run_compare(tmp_path, b)
    c = r["joints"]["little_cmc"]
    assert c["verdict"] == "unobservable"
    assert c["tracker_spread"]["max_from_mean_deg"] < 1e-6
    assert c["reference_range_deg"][1] == pytest.approx(12.0, abs=0.2)
    assert "mapping" not in c and "no mapping is fitted" in c["explanation"]
    rec = rc.load_recording(b["folder"] / b["takes"][0]["file"], "left")
    obs = rv.observability(rec)
    assert obs["THUMB_METACARPAL"]["max_from_mean_deg"] > 20.0
    for bone in rv.OBSERVED_BONES[1:]:
        assert obs[bone]["max_from_mean_deg"] < 1e-6


# --- split -----------------------------------------------------------------------------------
def test_split_the_fit_never_sees_a_held_out_sample(tmp_path):
    b = build(tmp_path, trials=3)
    r = run_compare(tmp_path, b)
    split = r["split"]
    assert split["calibration_trials"] == [1] and split["heldout_trials"] == [2, 3]
    assert not set(split["calibration_samples"]) & set(split["heldout_samples"])
    n_cal = sum(1 for m in r["_matches"] if m.used and m.sample.trial == 1)
    assert r["joints"]["thumb_tmc"]["mapping"]["n"] == n_cal
    rows = copy.deepcopy(b["rows"])
    for row in rows:                                     # spoil every held-out value
        if row["trial"] > 1 and row["tmc_fe"]:
            row["tmc_fe"] = f"{float(row['tmc_fe']) + 20.0:.9f}"
    r2 = run_compare(tmp_path / "x", b, rows=rows)
    a1 = np.array(r["joints"]["thumb_tmc"]["mapping"]["A"])
    a2 = np.array(r2["joints"]["thumb_tmc"]["mapping"]["A"])
    assert np.array_equal(a1, a2)                        # the fit did not move
    assert mapped_all(r2)["fe"]["bias_deg"] == pytest.approx(-20.0, abs=1e-6)
    rv.write_report(r, tmp_path / "out")
    m = json.loads((tmp_path / "out" / "mapping.json").read_text(encoding="utf-8"))
    assert m["calibration_trials"] == [1]
    assert set(m["calibration_samples"]).isdisjoint(m["heldout_samples"])


def test_split_refuses_a_split_with_nothing_held_out_or_nothing_to_fit(tmp_path):
    b = build(tmp_path)
    with pytest.raises(rv.ReferenceError, match="nothing is left to evaluate"):
        run_compare(tmp_path, b, trials=(1, 2))
    with pytest.raises(rv.ReferenceError, match="no calibration trial"):
        run_compare(tmp_path / "n", b, trials=(9,))
    with pytest.raises(rv.ReferenceError, match="no calibration trial"):
        run_compare(tmp_path / "e", b, trials=())


def test_split_a_calibration_trial_with_one_movement_is_refused_in_the_report(tmp_path):
    b = build(tmp_path)
    rows = [r for r in b["rows"]
            if not (r["trial"] == 1 and r["movement"] == "tmc_abduction_sweep")]
    r = run_compare(tmp_path, b, rows=rows)
    assert "do not determine the mapping" in r["joints"]["thumb_tmc"]["mapping"]["refused"]
    assert r["_mapping"] is None
