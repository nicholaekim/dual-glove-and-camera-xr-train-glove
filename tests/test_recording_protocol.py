"""cam_hand.recording_protocol: protocol files, schedules, rounds, events,
warm-up, fractions and the quick checks, all without hardware."""
import hashlib
import json
from pathlib import Path

import pytest

from cam_hand import recording_protocol as rp

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols"
ALL5 = ["thumb", "index", "middle", "ring", "pinky"]

# The seven sequences exactly as docs/protocol_formats.md spells them: the
# flexed set of every step, in order.
CONTRACT_SEQUENCES = {
    "seq1_one_at_a_time": [[], ["thumb"], ["index"], ["middle"], ["ring"],
                           ["pinky"], ALL5, []],
    "seq2_reverse": [[], ["pinky"], ["ring"], ["middle"], ["index"],
                     ["thumb"], ALL5, []],
    "seq3_alternating": [[], ["thumb"], ["middle"], ["index"], ["pinky"],
                         ["ring"], []],
    "seq4_pairs": [[], ["thumb", "index"], ["index", "middle"],
                   ["middle", "ring"], ["ring", "pinky"], ALL5, []],
    "seq5_progressive": [[], ["thumb"], ["thumb", "index"],
                         ["thumb", "index", "middle"],
                         ["thumb", "index", "middle", "ring"], ALL5, ALL5, []],
    "seq6_configurations": [[], ["index"], ["index", "middle"],
                            ["middle", "ring"], ["ring", "pinky"], ALL5, []],
    "seq7_flex_release": [[], ["thumb"], [], ["index"], [], ["middle"], [],
                          ["ring"], [], ["pinky"], []],
}
CONTRACT_LABELS = {
    "seq5_progressive": ["open hand", "thumb", "thumb + index",
                         "thumb + index + middle", "+ring", "+little",
                         "full fist", "open hand"],
    "seq7_flex_release": ["open hand", "thumb flex", "thumb extend",
                          "index flex", "index extend", "middle flex",
                          "middle extend", "ring flex", "ring extend",
                          "little flex", "little extend"],
}
GRASP_IDS = ["cylindrical", "spherical", "hook", "lateral_key", "tip_pinch",
             "tripod", "palmar_pinch", "extension_plate", "lateral_tripod",
             "power_sphere", "precision_disc", "writing_tripod"]


def load(name):
    return rp.load_protocol(PROTOCOLS / f"{name}.json")


def write(tmp_path, data, name="p.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- the protocol files --------------------------------------------------
def test_protocol_files_load_and_hash():
    for name in ("finger_flexion", "sequences", "grasps"):
        p = load(name)
        assert p.name == name and p.version == 1
        assert p.sha256 == hashlib.sha256(
            (PROTOCOLS / f"{name}.json").read_bytes()).hexdigest()
        assert isinstance(p.data["description"], str) and p.data["description"]


def test_finger_flexion_file_matches_the_contract():
    p = load("finger_flexion")
    assert p.takes_per_item == 1
    assert p.ids == ["thumb", "index", "middle", "ring", "pinky",
                     "index_slow", "index_fast"]
    for fid in ("thumb", "index", "middle", "ring", "pinky"):
        it = p.item(fid)
        assert it["finger"] == fid
        assert (it["cycles"], it["bend_s"], it["hold_s"], it["straighten_s"],
                it["rest_s"]) == (5, 4.0, 1.0, 4.0, 1.0)
    slow, fast = p.item("index_slow"), p.item("index_fast")
    assert (slow["finger"], slow["cycles"], slow["bend_s"], slow["hold_s"],
            slow["straighten_s"], slow["rest_s"]) == ("index", 5, 6.0, 1.0,
                                                      6.0, 1.0)
    assert (fast["finger"], fast["cycles"], fast["bend_s"], fast["hold_s"],
            fast["straighten_s"], fast["rest_s"]) == ("index", 8, 1.0, 0.5,
                                                      1.0, 0.5)
    assert slow["label"] == "index, slow" and fast["label"] == "index, fast"


def test_sequences_match_the_contract_step_by_step():
    p = load("sequences")
    assert p.data["hold_s"] == 2.5 and p.data["check_window_s"] == 1.5
    assert p.takes_per_item == 3 and p.shuffle_rounds is True
    assert p.ids == list(CONTRACT_SEQUENCES)
    for sid, flexed in CONTRACT_SEQUENCES.items():
        steps = p.item(sid)["steps"]
        assert [s["flexed"] for s in steps] == flexed, sid
        assert steps[0]["label"] == "open hand"
        for s in steps:
            # the professor's words in the labels, the data's name in flexed
            assert "pinky" not in s["label"]
    for sid, labels in CONTRACT_LABELS.items():
        assert [s["label"] for s in p.item(sid)["steps"]] == labels
    assert [len(p.item(s)["steps"]) for s in p.ids] == [8, 8, 7, 7, 8, 7, 11]
    assert any("little" in s["label"] for s in p.item("seq1_one_at_a_time")
               ["steps"])


def test_grasps_placeholder_set():
    p = load("grasps")
    assert p.data["status"] == "placeholder until the reference papers are in hand"
    assert p.takes_per_item == 3
    assert p.data["duration_s"] == 5.0 and p.data["prep_s"] == 5.0
    assert p.ids == GRASP_IDS
    for it in p.items:
        assert it["source"] is None and it["figure"] is None
        assert it["shape"].strip() and "\n" not in it["shape"]
        assert isinstance(it["object_implied"], bool)
    assert p.item("cylindrical")["object_implied"] is True
    assert p.item("tip_pinch")["object_implied"] is False


# --- validation ------------------------------------------------------------
def _seq_protocol(**over):
    data = {"name": "sequences", "version": 1, "description": "",
            "hold_s": 2.5, "check_window_s": 1.5, "takes_per_item": 1,
            "shuffle_rounds": False,
            "items": [{"id": "s1", "label": "S1",
                       "steps": [{"label": "open", "flexed": []},
                                 {"label": "idx", "flexed": ["index"]}]}]}
    data.update(over)
    return data


@pytest.mark.parametrize("change, message", [
    (dict(name="flexions"), "'name' must be one of"),
    (dict(version=0), "'version' must be a whole number"),
    (dict(version=True), "'version' must be a whole number"),
    (dict(items=[]), "'items' must be a non-empty list"),
    (dict(check_window_s=3.0), "cannot be longer than 'hold_s'"),
    (dict(hold_s=-1), "'hold_s' must be a number of seconds above 0"),
    (dict(shuffle_rounds="yes"), "'shuffle_rounds' must be true or false"),
    (dict(takes_per_item=0), "'takes_per_item' must be a whole number"),
])
def test_validation_names_the_problem(tmp_path, change, message):
    with pytest.raises(rp.ProtocolError, match=message):
        rp.load_protocol(write(tmp_path, _seq_protocol(**change)))


def test_validation_of_items(tmp_path):
    bad_finger = _seq_protocol()
    bad_finger["items"][0]["steps"][1]["flexed"] = ["little"]
    with pytest.raises(rp.ProtocolError, match="'pinky' in the data") as e:
        rp.load_protocol(write(tmp_path, bad_finger))
    assert "items[0] ('s1') step 1" in str(e.value)

    dup = _seq_protocol()
    dup["items"].append(dict(dup["items"][0]))
    with pytest.raises(rp.ProtocolError, match="used twice"):
        rp.load_protocol(write(tmp_path, dup))

    bad_id = _seq_protocol()
    bad_id["items"][0]["id"] = "Seq 1"
    with pytest.raises(rp.ProtocolError, match="file-name safe"):
        rp.load_protocol(write(tmp_path, bad_id))

    twice = _seq_protocol()
    twice["items"][0]["steps"][1]["flexed"] = ["index", "index"]
    with pytest.raises(rp.ProtocolError, match="names a finger twice"):
        rp.load_protocol(write(tmp_path, twice))

    flex = json.loads((PROTOCOLS / "finger_flexion.json").read_text())
    flex["items"][6]["cycles"] = 0
    with pytest.raises(rp.ProtocolError, match="'index_fast'.*'cycles'"):
        rp.load_protocol(write(tmp_path, flex))
    flex["items"][6]["cycles"] = 8
    flex["items"][6]["finger"] = "little"
    with pytest.raises(rp.ProtocolError, match="'finger' must be one of"):
        rp.load_protocol(write(tmp_path, flex))


def test_load_reports_json_position(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"name": "sequences",\n "version": }', encoding="utf-8")
    with pytest.raises(rp.ProtocolError, match="line 2"):
        rp.load_protocol(path)
    with pytest.raises(rp.ProtocolError, match="cannot be read"):
        rp.load_protocol(tmp_path / "missing.json")


def test_select_items():
    p = load("finger_flexion")
    assert rp.select_items(p) == p.ids
    assert rp.select_items(p, ["index_fast", "thumb"]) == ["thumb", "index_fast"]
    with pytest.raises(rp.ProtocolError, match="no item named nope"):
        rp.select_items(p, ["nope"])


# --- schedules ---------------------------------------------------------------
def test_flexion_schedules_for_every_item():
    p = load("finger_flexion")
    for fid in p.ids:
        it = p.item(fid)
        cues = rp.build_schedule(p, fid)
        assert len(cues) == it["cycles"] * 4
        assert [c.step for c in cues] == list(range(len(cues)))
        assert [c.phase for c in cues] == list(rp.PHASES) * it["cycles"]
        assert [c.cycle for c in cues] == [k for k in range(1, it["cycles"] + 1)
                                           for _ in range(4)]
        words = rp.FINGER_WORDS[it["finger"]]
        for c in cues:
            want = {"bend": (f"bend the {words}", (it["finger"],), it["bend_s"]),
                    "hold": ("hold", (it["finger"],), it["hold_s"]),
                    "straighten": ("straighten", (), it["straighten_s"]),
                    "rest": ("rest", (), it["rest_s"])}[c.phase]
            assert (c.label, c.flexed, c.hold_s) == want
        per_cycle = it["bend_s"] + it["hold_s"] + it["straighten_s"] + it["rest_s"]
        assert rp.schedule_seconds(cues) == pytest.approx(per_cycle * it["cycles"])
    assert len(rp.build_schedule(p, "index")) == 20
    assert len(rp.build_schedule(p, "index_fast")) == 32
    assert rp.build_schedule(p, "pinky")[0].label == "bend the little finger"


def test_sequence_schedules_for_all_seven():
    p = load("sequences")
    for sid, flexed in CONTRACT_SEQUENCES.items():
        cues = rp.build_schedule(p, sid)
        assert len(cues) == len(flexed)
        assert [list(c.flexed) for c in cues] == flexed
        assert [c.label for c in cues] == [s["label"] for s in p.item(sid)["steps"]]
        assert all(c.hold_s == 2.5 and c.cycle is None and c.phase is None
                   for c in cues)
        assert rp.cue_offsets(cues) == [2.5 * k for k in range(len(cues))]
    # flexed comes out in the hand's order whatever order the file used
    assert rp.sequence_cues({"steps": [{"label": "x", "flexed": ["pinky", "thumb"]}]},
                            1.0)[0].flexed == ("thumb", "pinky")


def test_time_scale_scales_every_duration():
    p = load("sequences")
    cues = rp.build_schedule(p, "seq3_alternating", time_scale=0.25)
    assert [c.hold_s for c in cues] == [0.625] * 7
    f = load("finger_flexion")
    fast = rp.build_schedule(f, "index_fast", time_scale=0.5)
    assert rp.schedule_seconds(fast) == pytest.approx(12.0)
    with pytest.raises(ValueError):
        rp.build_schedule(p, "seq3_alternating", time_scale=0)
    with pytest.raises(rp.ProtocolError, match="record_poses"):
        rp.build_schedule(load("grasps"), "cylindrical")


def test_event_fields_per_set():
    f = rp.build_schedule(load("finger_flexion"), "index")[0]
    assert f.event_fields() == {"step": 0, "label": "bend the index",
                                "flexed": ["index"], "hold_s": 4.0,
                                "cycle": 1, "phase": "bend"}
    s = rp.build_schedule(load("sequences"), "seq4_pairs")[1]
    assert s.event_fields() == {"step": 1, "label": "thumb + index",
                                "flexed": ["thumb", "index"], "hold_s": 2.5}


def test_cue_pitch():
    f = rp.build_schedule(load("finger_flexion"), "index")
    assert [rp.cue_pitch(c) for c in f[:4]] == [rp.PITCH_FLEX, rp.PITCH_HOLD,
                                                rp.PITCH_EXTEND, rp.PITCH_REST]
    seq = rp.build_schedule(load("sequences"), "seq5_progressive")
    pitches = [rp.cue_pitch(c, seq[k - 1] if k else None)
               for k, c in enumerate(seq)]
    assert pitches == [rp.PITCH_HOLD, rp.PITCH_FLEX, rp.PITCH_FLEX,
                       rp.PITCH_FLEX, rp.PITCH_FLEX, rp.PITCH_FLEX,
                       rp.PITCH_HOLD, rp.PITCH_EXTEND]
    pairs = rp.build_schedule(load("sequences"), "seq4_pairs")
    assert rp.cue_pitch(pairs[3], pairs[2]) == rp.PITCH_FLEX   # ring comes in
    release = rp.build_schedule(load("sequences"), "seq7_flex_release")
    assert rp.cue_pitch(release[2], release[1]) == rp.PITCH_EXTEND


def test_still_cue():
    fast = rp.build_schedule(load("finger_flexion"), "index_fast")
    k = rp.still_cue(fast)
    assert fast[k].phase == "hold" and fast[k].cycle == 4
    five = rp.build_schedule(load("finger_flexion"), "index")
    assert five[rp.still_cue(five)].cycle == 3
    seq = rp.build_schedule(load("sequences"), "seq3_alternating")
    assert rp.still_cue(seq) == 3


# --- rounds ------------------------------------------------------------------
def test_rounds_unshuffled_keep_the_protocol_order():
    assert rp.rounds(["a", "b", "c"], 2) == [["a", "b", "c"], ["a", "b", "c"]]


def test_rounds_are_deterministic_and_cover_every_item():
    ids = load("sequences").ids
    one = rp.rounds(ids, 3, seed=12345, shuffle=True)
    assert one == rp.rounds(ids, 3, seed=12345, shuffle=True)
    assert len(one) == 3
    assert all(sorted(r) == sorted(ids) for r in one)
    assert any(rp.rounds(ids, 3, seed=s, shuffle=True) != one
               for s in range(1, 6))
    for seed in range(200):
        plan = rp.rounds(ids, 3, seed=seed, shuffle=True)
        assert all(sorted(r) == sorted(ids) for r in plan)
        for a, b in zip(plan, plan[1:]):
            assert a[-1] != b[0], seed     # never the same take back to back
    assert rp.rounds(["only"], 3, seed=1, shuffle=True) == [["only"]] * 3


def test_rounds_refuse_nonsense():
    with pytest.raises(ValueError):
        rp.rounds([], 3)
    with pytest.raises(ValueError):
        rp.rounds(["a", "a"], 1)
    with pytest.raises(ValueError):
        rp.rounds(["a"], 0)


def test_take_name():
    assert rp.take_name("index_fast", "left", 2, "20260928_140530") == \
        "index_fast_left_take2_20260928_140530"


# --- events ------------------------------------------------------------------
def test_events_round_trip(tmp_path):
    path = tmp_path / "events" / "x.events.jsonl"
    cues = rp.build_schedule(load("finger_flexion"), "index")[:2]
    with rp.EventLog(path) as log:
        a = log.write("take_start", t=100.0, item="index", take=1)
        b = log.write("cue", t=100.5, **cues[0].event_fields())
        c = log.write("cue", t=104.5, **cues[1].event_fields())
        d = log.write("take_end", t=105.5, item="index", take=1)
    e = rp.append_event(path, "decision", t=106.0, accepted=True, reason="",
                        by="auto")
    rows = rp.read_events(path)
    assert rows == [a, b, c, d, e]
    assert rows[0] == {"t": 100.0, "kind": "take_start", "item": "index",
                       "take": 1}
    assert rows[1]["phase"] == "bend" and rows[1]["cycle"] == 1
    assert rp.take_window(rows) == (100.0, 105.5)
    assert [r["step"] for r in rp.cue_events(rows)] == [0, 1]
    with rp.EventLog(tmp_path / "now.jsonl") as log:
        row = log.write("cue")
    assert row["t"] > 1.7e9


def test_read_events_names_a_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"t": 1, "kind": "cue"}\n\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        rp.read_events(path)
    path.write_text('{"kind": "cue"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="needs 't'"):
        rp.read_events(path)


# --- fractions and the warm-up -------------------------------------------------
def test_fraction_and_its_clip():
    assert rp.fraction(2.0, 1.0, 2.0) == 0.0
    assert rp.fraction(2.0, 1.0, 1.0) == 1.0
    assert rp.fraction(2.0, 1.0, 1.5) == pytest.approx(0.5)
    assert rp.fraction(2.0, 1.0, 3.0) == -0.5       # clipped from -1.0
    assert rp.fraction(2.0, 1.0, 0.0) == 1.5        # clipped from 2.0
    assert rp.fraction(2.0, 1.0, 2.4) == pytest.approx(-0.4)
    assert rp.fraction(1.0, 1.0, 0.5) is None       # no range at all
    assert rp.fraction(1.0, 2.0, 1.5) == pytest.approx(0.5)   # inverted glove


OPEN = [1.40, 1.97, 2.07, 1.97, 1.71]
FIST = [0.95, 0.66, 0.62, 0.66, 0.76]


def curls_at(fracs):
    """Five fractions -> five curls on the OPEN/FIST endpoints above."""
    return [o - f * (o - k) for o, k, f in zip(OPEN, FIST, fracs)]


def glove_ends(open_=OPEN, fist=FIST):
    return {"open": dict(zip(ALL5, open_)), "fist": dict(zip(ALL5, fist)),
            "span": {f: o - k for f, o, k in zip(ALL5, open_, fist)},
            "frames": 100}


def test_warmup_medians_span_and_windows():
    samples = []
    for i in range(60):                                  # open window 0..3 s
        t = i * 0.05
        c = list(OPEN)
        if i < 5:
            c = curls_at([0.9] * 5)                      # still moving in
        c[1] += 0.01 * (i % 3 - 1)
        samples.append((t, c))
    for i in range(60):                                  # fist window 3..6 s
        samples.append((3.0 + i * 0.05, list(FIST)))
    samples.append((10.0, [9.9] * 5))                    # after both: ignored
    ends = rp.window_endpoints(samples, (1.0, 2.99), (4.0, 6.0))
    assert ends["open"]["index"] == pytest.approx(1.97, abs=1e-3)
    assert ends["fist"]["ring"] == pytest.approx(0.66)
    assert ends["span"]["middle"] == pytest.approx(2.07 - 0.62, abs=1e-4)
    assert ends["frames"] == 40 + 40
    assert rp.window_endpoints(samples, (20.0, 21.0), (4.0, 6.0)) is None

    rec = rp.warmup_record("left", (1.0, 2.99), (4.0, 6.0), samples,
                           camera_samples=None, settle_s=1.0)
    assert set(rec) == {"hand", "t_open", "t_fist", "glove", "camera",
                        "refused", "settle_s"}
    assert rec["hand"] == "left" and rec["camera"] is None
    assert rec["refused"] is None
    rec = rp.warmup_record("left", (1.0, 2.99), (4.0, 6.0), samples, [])
    assert rec["camera"] is None            # a camera that saw nothing


def test_warmup_refusal_rule():
    assert rp.warmup_refusal(glove_ends()) is None
    weak_thumb = glove_ends(fist=[1.35] + FIST[1:])       # thumb span 0.05
    assert rp.warmup_refusal(weak_thumb) is None          # thumb never refuses
    weak_ring = glove_ends(fist=FIST[:3] + [1.75] + FIST[4:])
    why = rp.warmup_refusal(weak_ring)
    assert "ring 0.22" in why and "0.30" in why
    assert rp.warmup_refusal(None).startswith("the glove sent no frames")
    samples = [(0.5, OPEN), (1.5, curls_at([0.9, 0.2, 0.9, 0.9, 0.9]))]
    rec = rp.warmup_record("right", (0, 1), (1, 2), samples)
    assert rec["refused"] and "index 0.26" in rec["refused"]


# --- peaks and the Set B check --------------------------------------------------
def flexion_trace(finger="index", cycles=5, amplitude=1.0, dt=1 / 60.0,
                  bend=1.0, hold=0.5, straighten=1.0, rest=0.5, t0=100.0):
    """A glove trace of one finger doing `cycles` cycles, events to match."""
    i = ALL5.index(finger)
    events = [{"t": t0, "kind": "take_start", "item": finger, "take": 1}]
    curls, t, step = [], t0, 0
    for cycle in range(1, cycles + 1):
        for phase, dur in (("bend", bend), ("hold", hold),
                           ("straighten", straighten), ("rest", rest)):
            events.append({"t": t, "kind": "cue", "step": step, "label": phase,
                           "flexed": [finger] if phase in ("bend", "hold") else [],
                           "hold_s": dur, "cycle": cycle, "phase": phase})
            step += 1
            n = int(round(dur / dt))
            for k in range(n):
                e = k / n
                f = {"bend": e, "hold": 1.0, "straighten": 1.0 - e,
                     "rest": 0.0}[phase] * amplitude
                fr = [0.0] * 5
                fr[i] = f
                curls.append((t + k * dt, curls_at(fr)))
            t += dur
    events.append({"t": t, "kind": "take_end", "item": finger, "take": 1})
    return curls, events


def test_count_peaks():
    curls, _ = flexion_trace(cycles=5)
    fr = [rp.fraction(OPEN[1], FIST[1], c[1]) for _t, c in curls]
    assert rp.count_peaks(fr) == 5
    assert rp.count_peaks([0.01, 0.02, 0.0, 0.03] * 20) == 0     # no range
    assert rp.count_peaks([1.0, 1.0, 0.0, 0.0, 1.0, 0.0]) == 1   # starts bent
    assert rp.count_peaks([]) == 0


def test_flexion_check_accepts_a_full_take():
    curls, events = flexion_trace(cycles=5)
    res = rp.check_flexion_take(curls, events, glove_ends(), "index", 5)
    assert res.accepted and res.reason == ""
    assert res.details["span_fraction"] == pytest.approx(1.0, abs=0.02)
    assert res.details["cycles_from_events"] == 5
    assert res.details["cycles_from_peaks"] == 5
    assert res.details["cycles_match"] is True
    assert res.details["other_spans"]["middle"] == pytest.approx(0.0)
    assert res.notes == []
    assert res.as_dict()["accepted"] is True


def test_flexion_check_rejects_naming_finger_and_fraction():
    curls, events = flexion_trace(finger="ring", cycles=5, amplitude=0.4)
    res = rp.check_flexion_take(curls, events, glove_ends(), "ring", 5)
    assert not res.accepted
    assert res.reason == ("the ring moved only 0.40 of its warm-up range "
                          "(needs 0.60)")
    pinky, events = flexion_trace(finger="pinky", cycles=5, amplitude=0.1)
    res = rp.check_flexion_take(pinky, events, glove_ends(), "pinky", 5)
    assert "the little finger moved only 0.10" in res.reason


def test_flexion_check_flags_a_cycle_mismatch_without_rejecting():
    curls, events = flexion_trace(cycles=4)
    res = rp.check_flexion_take(curls, events, glove_ends(), "index", 5)
    assert res.accepted
    assert res.details["cycles_match"] is False
    assert any("cycles: 5 cued in the protocol, 4 in the events, 4 peaks"
               in n for n in res.notes)


def test_flexion_check_with_no_frames_or_no_range():
    _curls, events = flexion_trace(cycles=2)
    res = rp.check_flexion_take([], events, glove_ends(), "index", 2)
    assert not res.accepted and "no glove frames" in res.reason
    flat = glove_ends(fist=[1.40] + FIST[1:])
    curls, events = flexion_trace(finger="thumb", cycles=2)
    res = rp.check_flexion_take(curls, events, flat, "thumb", 2)
    assert not res.accepted and "no warm-up range" in res.reason


# --- the Set C check ------------------------------------------------------------
def sequence_trace(steps, hold=2.5, dt=1 / 60.0, t0=50.0, override=None):
    """A glove that reaches every step's set within 0.6 s and holds it.

    `override` = {(step, finger): fraction} replaces what one finger reads
    during that step's hold.
    """
    override = override or {}
    events = [{"t": t0, "kind": "take_start", "item": "s", "take": 1}]
    curls, prev = [], [0.0] * 5
    for k, flexed in enumerate(steps):
        t = t0 + k * hold
        events.append({"t": t, "kind": "cue", "step": k, "label": f"s{k}",
                       "flexed": flexed, "hold_s": hold})
        target = [1.0 if f in flexed else 0.0 for f in ALL5]
        for i, f in enumerate(ALL5):
            if (k, f) in override:
                target[i] = override[(k, f)]
        n = int(round(hold / dt))
        for j in range(n):
            a = min(1.0, (j * dt) / 0.6)
            fr = [p + (g - p) * a for p, g in zip(prev, target)]
            curls.append((t + j * dt, curls_at(fr)))
        prev = target
    events.append({"t": t0 + len(steps) * hold, "kind": "take_end",
                   "item": "s", "take": 1})
    return curls, events


def test_step_windows_end_where_the_step_really_ended():
    events = [{"t": 0.0, "kind": "take_start"},
              {"t": 1.0, "kind": "cue", "step": 0, "hold_s": 2.5},
              {"t": 3.6, "kind": "cue", "step": 1, "hold_s": 2.5},
              {"t": 6.0, "kind": "take_end"}]
    w = rp.step_windows(events, 1.5)
    assert [(round(a, 3), round(b, 3)) for _c, a, b in w] == [(2.1, 3.6),
                                                             (4.5, 6.0)]
    short = rp.step_windows(events, 5.0)
    assert short[0][1] == 1.0                     # never before its own cue


def test_sequence_check_accepts_every_contract_sequence():
    for sid, steps in CONTRACT_SEQUENCES.items():
        curls, events = sequence_trace(steps)
        res = rp.check_sequence_take(curls, events, glove_ends(), 1.5)
        assert res.accepted, (sid, res.reason)
        assert res.details["steps_total"] == len(steps)
        assert res.details["steps_pass"] == len(steps)
        assert res.notes == []


def test_sequence_check_reports_coupling_without_rejecting():
    steps = CONTRACT_SEQUENCES["seq1_one_at_a_time"]
    curls, events = sequence_trace(steps, override={(4, "middle"): 0.45,
                                                    (4, "pinky"): 0.35})
    res = rp.check_sequence_take(curls, events, glove_ends(), 1.5)
    assert res.accepted
    assert "step 4 (s4): middle 0.45, little finger 0.35" in res.notes[0]
    step4 = res.details["steps"][4]
    assert step4["pass"] and step4["coupling"] == ["middle 0.45",
                                                  "little finger 0.35"]
    assert step4["fractions"]["ring"] == pytest.approx(1.0, abs=1e-3)


def test_sequence_check_rejects_a_flexed_finger_that_failed():
    steps = CONTRACT_SEQUENCES["seq3_alternating"]
    curls, events = sequence_trace(steps, override={(2, "middle"): 0.4})
    res = rp.check_sequence_take(curls, events, glove_ends(), 1.5)
    assert not res.accepted
    assert res.reason == ("step 2 (s2): middle read 0.40 of its range, should "
                          "be flexed (above 0.60)")
    assert res.details["steps_pass"] == len(steps) - 1


def test_sequence_check_rejects_a_straight_finger_that_reads_flexed():
    steps = CONTRACT_SEQUENCES["seq4_pairs"]
    curls, events = sequence_trace(steps, override={(1, "ring"): 0.8,
                                                    (3, "thumb"): 0.9})
    res = rp.check_sequence_take(curls, events, glove_ends(), 1.5)
    assert not res.accepted
    assert res.reason == ("step 1 (s1): ring read flexed (0.80) but should be "
                          "straight; 1 more step(s) failed")


def test_sequence_check_with_no_frames():
    steps = [[], ["index"]]
    _curls, events = sequence_trace(steps)
    res = rp.check_sequence_take([], events, glove_ends(), 1.5)
    assert not res.accepted
    assert res.reason.startswith("step 0 (s0): no glove frames in the hold "
                                 "window")
    res = rp.check_sequence_take([], [], glove_ends(), 1.5)
    assert not res.accepted and "no cues" in res.reason


def test_check_take_dispatches_on_the_set():
    p = load("sequences")
    steps = CONTRACT_SEQUENCES["seq3_alternating"]
    curls, events = sequence_trace(steps, hold=0.625)
    assert rp.check_take(p, "seq3_alternating", curls, events, glove_ends(),
                         time_scale=0.25).accepted
    f = load("finger_flexion")
    curls, events = flexion_trace(cycles=8)
    res = rp.check_take(f, "index_fast", curls, events, glove_ends())
    assert res.accepted and res.details["cycles_from_peaks"] == 8


# --- reading a take back ----------------------------------------------------------
def test_read_curls_uses_capture_time(tmp_path):
    from xr_hand.mock import MockHandGenerator
    from xr_hand.parser import parse_hand_message
    from xr_hand.recorder import FrameRecorder

    path = tmp_path / "g.jsonl"
    rec = FrameRecorder(take=1)
    rec.start(path)
    gl, gr = MockHandGenerator("left"), MockHandGenerator("right")
    for _ in range(5):
        rec.record(parse_hand_message(gl.next_frame(), "left"))
        rec.record(parse_hand_message(gr.next_frame(), "right"))
    rec.stop()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for k, row in enumerate(rows):
        row["capture_time"] = 1000.0 + k
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    left = rp.read_curls(path, "left")
    assert [t for t, _c in left] == [1000.0, 1002.0, 1004.0, 1006.0, 1008.0]
    assert all(len(c) == 5 for _t, c in left)
    assert len(rp.read_curls(path)) == 10
    assert rp.read_curls(tmp_path / "none.jsonl") == []
