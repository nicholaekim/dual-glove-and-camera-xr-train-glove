"""The protocol checker, on synthetic sessions whose answers are known.

Every session here is written in the contract's layout
(`docs/protocol_formats.md`): session.json, warmup.json, glove and camera
frames in the real line format (the glove's `_frame_to_dict` line, the
camera's `LeapRecorder` line with `abs26`, `palm_abs` and the capture facts),
the three added keys, an events file, and rejected attempts under
`rejected/`. The hands come from the repo's mocks: the glove from
`xr_hand.mock.MockHandGenerator` with each finger's phalanges bent on its own,
the grasps' camera from `leap_hand.mock.MockLeapStream`, the flexion camera
from the same glove skeleton turned palm-down over the module and passed
through `leap_hand.to_openxr.to_hand_frame`. So each finger's curl is set
exactly, the glove trails the camera by a known lag, and a glove fault is a
fault put there on purpose.

`test_package_professor_set.py` imports the builders below, which is why they
are plain functions and not fixtures.
"""
import csv
import hashlib
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from cam_hand import protocol_check as pc
from cam_hand.features import flexion_features
from leap_hand.mock import MockLeapStream
from leap_hand.to_openxr import to_hand_frame
from leap_hand.types import LeapHand
from xr_hand.joints import HandFrame
from xr_hand.keypoints21 import frame_to_keypoints21
from xr_hand.kinematics import forward_kinematics_full, mat3_to_quat
from xr_hand.mock import MockHandGenerator, _rotx_quat
from xr_hand.parser import parse_hand_message
from xr_hand.recorder import _frame_to_dict

REPO = Path(__file__).resolve().parents[1]
FINGERS = pc.FINGERS
ALL = list(FINGERS)
PREFIX = {"thumb": "THUMB_", "index": "INDEX_", "middle": "MIDDLE_",
          "ring": "RING_", "pinky": "LITTLE_"}
FIST_ANGLE = 1.2           # rad per phalanx: the mock's full fist
GLOVE_HZ = 30.0
CAM_HZ = 40.0
LAG_S = 0.10               # the glove trails the camera by this much
# Palm-down over the module: glove +Y (fingertips) -> leap -z, glove +Z ->
# leap +y, so the palm (glove -Z) faces the lens.
R_CAM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
WRIST_CAM = {"left": np.array([-0.03, 0.21, 0.05]),
             "right": np.array([0.03, 0.21, 0.05])}

FLEXION_ITEMS = [
    {"id": "index", "label": "index", "finger": "index", "cycles": 5,
     "bend_s": 1.0, "hold_s": 0.5, "straighten_s": 1.0, "rest_s": 0.5},
    {"id": "middle", "label": "middle", "finger": "middle", "cycles": 5,
     "bend_s": 1.0, "hold_s": 0.5, "straighten_s": 1.0, "rest_s": 0.5},
    {"id": "ring", "label": "ring", "finger": "ring", "cycles": 5,
     "bend_s": 1.0, "hold_s": 0.5, "straighten_s": 1.0, "rest_s": 0.5},
]
SEQ4_STEPS = [
    {"label": "open hand", "flexed": []},
    {"label": "thumb + index", "flexed": ["thumb", "index"]},
    {"label": "index + middle", "flexed": ["index", "middle"]},
    {"label": "middle + ring", "flexed": ["middle", "ring"]},
    {"label": "ring + little", "flexed": ["ring", "pinky"]},
    {"label": "all fingers flexed", "flexed": ALL},
    {"label": "open hand", "flexed": []},
]
SEQUENCE_PROTOCOL = {"name": "sequences", "version": 1, "description": "test",
                     "hold_s": 2.5, "check_window_s": 1.5,
                     "takes_per_item": 3, "shuffle_rounds": True,
                     "items": [{"id": "seq4_pairs", "label": "pairs",
                                "steps": SEQ4_STEPS}]}
FLEXION_PROTOCOL = {"name": "finger_flexion", "version": 1,
                    "description": "test", "takes_per_item": 1,
                    "items": FLEXION_ITEMS}


# --- the synthetic hand --------------------------------------------------------

@lru_cache(maxsize=2)
def base_frame(hand: str) -> HandFrame:
    return parse_hand_message(MockHandGenerator(hand).next_frame(), hand)


def hand_frame(hand: str, angles: Dict[str, float], counter: int) -> HandFrame:
    """The mock glove hand with each finger's phalanges bent by its own angle."""
    joints = []
    for j in base_frame(hand).joints:
        finger = next((f for f, p in PREFIX.items() if j.name.startswith(p)),
                      None)
        if finger and not j.name.endswith("METACARPAL"):
            qx, qy, qz, qw = _rotx_quat(-angles.get(finger, 0.0))
            j = replace(j, qx=qx, qy=qy, qz=qz, qw=qw)
        joints.append(j)
    return HandFrame(timestamp=float(counter), packet_counter=counter,
                     hand_side=hand, frame_id=counter, status=1, joints=joints)


@lru_cache(maxsize=2)
def curl_table(hand: str):
    angles = np.linspace(0.0, FIST_ANGLE, 121)
    curls = np.asarray([flexion_features(frame_to_keypoints21(
        hand_frame(hand, {f: a for f in FINGERS}, 0))) for a in angles])
    return angles, curls


def curls_at(hand: str, frac: float) -> Dict[str, float]:
    """Each finger's curl when every finger sits at `frac` of the range."""
    a = min(max(frac, 0.0), 1.0) * FIST_ANGLE
    c = flexion_features(frame_to_keypoints21(
        hand_frame(hand, {f: a for f in FINGERS}, 0)))
    return dict(zip(FINGERS, c))


def angles_for(hand: str, fracs: Dict[str, float]) -> Dict[str, float]:
    """Target fraction of the warm-up range -> phalanx angle, per finger."""
    angles, curls = curl_table(hand)
    out = {}
    for i, f in enumerate(FINGERS):
        opn, fst = curls[0, i], curls[-1, i]
        target = opn - min(max(fracs.get(f, 0.0), 0.0), 1.0) * (opn - fst)
        out[f] = float(np.interp(target, curls[::-1, i], angles[::-1]))
    return out


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# --- line writers in the recorders' format -----------------------------------

def glove_line(hand, fracs, t, counter, tags) -> dict:
    d = _frame_to_dict(hand_frame(hand, angles_for(hand, fracs), counter),
                       wall_time=t + 0.002)
    d["capture_time"] = round(t, 6)
    d.update(tags)
    return d


def camera_line(hand, fracs, t, counter, tags, rng, label=None,
                hand_id=2001, visible_us=None) -> dict:
    """One LeapRecorder line of the same skeleton, palm-down over the module."""
    fr = hand_frame(hand, angles_for(hand, fracs), counter)
    pos, rots = forward_kinematics_full(fr)
    wrist = np.asarray(pos[1])
    abs26 = [R_CAM @ (np.asarray(p) - wrist) + WRIST_CAM[hand]
             + rng.normal(0.0, 0.0002, 3) for p in pos]
    quats = [list(mat3_to_quat(R_CAM @ r)) for r in rots]
    lh = LeapHand(hand_side=label or hand, hand_id=hand_id,
                  timestamp_us=int(t * 1e6), frame_id=counter, framerate=CAM_HZ,
                  visible_time_us=int(visible_us if visible_us is not None
                                      else 2e6 + counter * 1e6 / CAM_HZ),
                  pinch_strength=0.1,
                  grab_strength=round(float(np.mean(list(fracs.values()))), 3),
                  palm_pos=[float(v) for v in abs26[0]],
                  palm_quat=quats[0],
                  abs26=[[float(v) for v in a] for a in abs26], quat26=quats,
                  frame_age_us=None, capture_time=t)
    return leap_line(lh, t + 0.004, tags)


def leap_line(lh: LeapHand, wall_time: float, tags) -> dict:
    """Exactly what `leap_hand.recorder.LeapRecorder.record` writes."""
    d = _frame_to_dict(to_hand_frame(lh), wall_time=wall_time)
    d.update({
        "source": "leap", "space": "leap_desktop", "units": "m",
        "timestamp_us": lh.timestamp_us, "hand_id": lh.hand_id,
        "visible_time_us": lh.visible_time_us,
        "framerate": round(lh.framerate, 3),
        "pinch_strength": lh.pinch_strength,
        "grab_strength": lh.grab_strength, "frame_age_us": None,
        "capture_time": round(lh.capture_time, 6),
        "palm_abs": [round(float(c), 6) for c in lh.palm_pos],
        "abs26": [[round(float(c), 6) for c in p] for p in lh.abs26],
    })
    d.update(tags)
    return d


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --- whole sessions ----------------------------------------------------------------

@dataclass
class FlexTake:
    item: str
    peaks: Optional[List[float]] = None       # per cycle, glove and camera
    coupling: Dict[str, float] = field(default_factory=dict)
    accepted: bool = True
    reason: str = ""
    camera_follows: bool = True
    cycles: Optional[int] = None              # cues written (default: item's)
    camera_label: Optional[str] = None        # tracker's label if not the hand


@dataclass
class SeqTake:
    item: str = "seq4_pairs"
    steps: List[dict] = field(default_factory=lambda: list(SEQ4_STEPS))
    glove_override: Dict[int, Dict[str, float]] = field(default_factory=dict)
    camera_override: Dict[int, Dict[str, float]] = field(default_factory=dict)
    glove_scale: float = 1.0                  # < 1: the glove reads too open
    accepted: bool = True
    reason: str = ""


def _stamp(k: int) -> str:
    return f"20260928_14{k // 60:02d}{k % 60:02d}"


def write_protocol(folder: Path, proto: dict) -> Path:
    path = folder / "protocols" / f"{proto['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proto, indent=1), encoding="utf-8")
    return path


def _session_dir(root: Path, kind: str, hand: str) -> Path:
    d = root / "recordings" / "protocol" / kind / f"20260928_140000_{hand}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_warmup(session: Path, hand: str, camera: bool, t0: float) -> None:
    opn, fst = curls_at(hand, 0.0), curls_at(hand, 1.0)
    block = {"open": opn, "fist": fst,
             "span": {f: opn[f] - fst[f] for f in FINGERS}, "frames": 178}
    (session / "warmup.json").write_text(json.dumps({
        "hand": hand, "t_open": [t0, t0 + 3], "t_fist": [t0 + 3, t0 + 6],
        "glove": block,
        "camera": dict(block, frames=240) if camera else None,
        "refused": None}, indent=1), encoding="utf-8")


def write_session_json(session: Path, kind: str, hand: str, proto_path: Path,
                       camera: bool, takes: List[dict], operator="N Kim"
                       ) -> None:
    data = proto_path.read_bytes()
    (session / "session.json").write_text(json.dumps({
        "set": kind, "protocol_file": str(proto_path),
        "protocol_name": kind, "protocol_version": 1,
        "protocol_sha256": hashlib.sha256(data).hexdigest(),
        "hand": hand, "operator": operator,
        "camera": "leap" if camera else "none", "glove": kind != "grasps",
        "started": "2026-09-28T14:00:00", "ended": "2026-09-28T14:40:00",
        "xr_trainer_calibrated_at": "2026-09-28T13:58:00" if kind != "grasps"
        else None, "seed": None, "rounds": None, "tool_commit": None,
        "takes": takes}, indent=1), encoding="utf-8")


def _take_entry(session: Path, name, item, n, accepted, reason, t, files,
                ) -> dict:
    base = "" if accepted else "rejected/"
    if not accepted:
        (session / "rejected").mkdir(exist_ok=True)
        (session / "rejected" / f"{name}.reason.txt").write_text(
            reason, encoding="utf-8")
    return {"item": item, "take": n, "name": name, "accepted": accepted,
            "reason": reason, "decided_at": t,
            "files": {k: base + v for k, v in files.items()}}


def _files(name: str, camera: bool, stills: bool) -> dict:
    files = {"glove": f"glove/{name}.jsonl",
             "events": f"events/{name}.events.jsonl"}
    if camera:
        files["leap"] = f"leap/{name}.jsonl"
    if camera and stills:
        files["still"] = f"stills/{name}.png"
    return files


def make_flexion_session(root: Path, takes: List[FlexTake], hand="left",
                         camera=True, stills=True, protocol=None) -> Path:
    """A finger_flexion session: 5 paced cycles per take, glove trailing."""
    session = _session_dir(root, "finger_flexion", hand)
    proto_path = write_protocol(root, protocol or FLEXION_PROTOCOL)
    items = {it["id"]: it for it in (protocol or FLEXION_PROTOCOL)["items"]}
    t = 1_790_000_000.0
    write_warmup(session, hand, camera, t)
    t += 10.0
    rng = np.random.default_rng(3)
    entries, counters, numbers = [], {"g": 0, "c": 0}, {}
    for k, spec in enumerate(takes):
        it = items[spec.item]
        finger = it["finger"]
        n = numbers.get(spec.item, 0) + 1
        if spec.accepted:
            numbers[spec.item] = n
        name = f"{spec.item}_{hand}_take{n}_{_stamp(k)}"
        base = session if spec.accepted else session / "rejected"
        tags = {"session": session.name, "item": spec.item, "take": n}
        cycles = spec.cycles or it["cycles"]
        peaks = spec.peaks or [1.0] * cycles
        start = t + 0.5
        period = it["bend_s"] + it["hold_s"] + it["straighten_s"] + it["rest_s"]
        events = [{"t": t, "kind": "take_start", "item": spec.item, "take": n}]
        step = 0
        for c in range(cycles):
            c0 = start + c * period
            for phase, dur, flexed in (
                    ("bend", it["bend_s"], [finger]),
                    ("hold", it["hold_s"], [finger]),
                    ("straighten", it["straighten_s"], []),
                    ("rest", it["rest_s"], [])):
                events.append({"t": c0, "kind": "cue", "step": step,
                               "label": f"{phase} the {finger}",
                               "flexed": flexed, "hold_s": dur,
                               "cycle": c + 1, "phase": phase})
                step += 1
                c0 += dur
        t_end = start + cycles * period + 0.3

        def state(tt, peaks=peaks, it=it, start=start, period=period,
                  cycles=cycles):
            x = tt - start
            if x < 0 or x >= cycles * period:
                return 0.0
            c, u = int(x // period), x % period
            p = peaks[c] if c < len(peaks) else 1.0
            if u < it["bend_s"]:
                return p * smooth(u / it["bend_s"])
            u -= it["bend_s"]
            if u < it["hold_s"]:
                return p
            u -= it["hold_s"]
            if u < it["straighten_s"]:
                return p * (1.0 - smooth(u / it["straighten_s"]))
            return 0.0

        def fracs(tt, spec=spec, finger=finger, state=state, cam=False):
            v = state(tt)
            out = {f: spec.coupling.get(f, 0.0) * v for f in FINGERS}
            out[finger] = 0.0 if (cam and not spec.camera_follows) else v
            if cam and not spec.camera_follows:
                out = {f: 0.05 for f in FINGERS}
            return out

        grows = []
        for i in range(int((t_end - t) * GLOVE_HZ)):
            tt = t + i / GLOVE_HZ
            counters["g"] += 1
            grows.append(glove_line(hand, fracs(tt - LAG_S), tt,
                                    counters["g"], tags))
        write_jsonl(base / "glove" / f"{name}.jsonl", grows)
        if camera:
            crows = []
            for i in range(int((t_end - t) * CAM_HZ)):
                tt = t + i / CAM_HZ
                counters["c"] += 1
                crows.append(camera_line(hand, fracs(tt, cam=True), tt,
                                         counters["c"], tags, rng,
                                         label=spec.camera_label))
            write_jsonl(base / "leap" / f"{name}.jsonl", crows)
            if stills:
                (base / "stills").mkdir(parents=True, exist_ok=True)
                (base / "stills" / f"{name}.png").write_bytes(b"\x89PNG fake")
        events.append({"t": t_end, "kind": "take_end", "item": spec.item,
                       "take": n})
        events.append({"t": t_end + 1.0, "kind": "decision",
                       "accepted": spec.accepted, "reason": spec.reason,
                       "by": "auto"})
        write_jsonl(base / "events" / f"{name}.events.jsonl", events)
        entries.append(_take_entry(session, name, spec.item, n, spec.accepted,
                                   spec.reason, t_end + 1.0,
                                   _files(name, camera, stills)))
        t = t_end + 5.0
    write_session_json(session, "finger_flexion", hand, proto_path, camera,
                       entries)
    return session


def make_sequence_session(root: Path, takes: List[SeqTake], hand="left",
                          camera=True, stills=True) -> Path:
    """A sequences session: each step cued, moved in 0.6 s, held 2.5 s."""
    session = _session_dir(root, "sequences", hand)
    proto_path = write_protocol(root, SEQUENCE_PROTOCOL)
    hold = SEQUENCE_PROTOCOL["hold_s"]
    t = 1_790_100_000.0
    write_warmup(session, hand, camera, t)
    t += 10.0
    rng = np.random.default_rng(5)
    entries, counters, numbers = [], {"g": 0, "c": 0}, {}
    for k, spec in enumerate(takes):
        n = numbers.get(spec.item, 0) + 1
        if spec.accepted:
            numbers[spec.item] = n
        name = f"{spec.item}_{hand}_take{n}_{_stamp(k)}"
        base = session if spec.accepted else session / "rejected"
        tags = {"session": session.name, "item": spec.item, "take": n}
        start = t + 0.5
        events = [{"t": t, "kind": "take_start", "item": spec.item, "take": n}]
        targets = {GLOVE: [], CAMERA: []}
        for s, st in enumerate(spec.steps):
            events.append({"t": start + s * hold, "kind": "cue", "step": s,
                           "label": st["label"], "flexed": st["flexed"],
                           "hold_s": hold})
            base_t = {f: (1.0 if f in st["flexed"] else 0.0) for f in FINGERS}
            g = dict(base_t, **spec.glove_override.get(s, {}))
            targets[GLOVE].append({f: v * spec.glove_scale
                                   for f, v in g.items()})
            targets[CAMERA].append(dict(base_t,
                                        **spec.camera_override.get(s, {})))
        t_end = start + len(spec.steps) * hold + 0.2

        def fracs(tt, sensor, targets=targets, start=start, n_steps=len(
                spec.steps)):
            x = tt - start
            if x < 0:
                return {f: 0.0 for f in FINGERS}
            s = min(int(x // hold), n_steps - 1)
            prev = targets[sensor][s - 1] if s > 0 else {f: 0.0
                                                         for f in FINGERS}
            w = smooth((x - s * hold) / 0.6)
            cur = targets[sensor][s]
            return {f: prev[f] + w * (cur[f] - prev[f]) for f in FINGERS}

        grows = []
        for i in range(int((t_end - t) * GLOVE_HZ)):
            tt = t + i / GLOVE_HZ
            counters["g"] += 1
            grows.append(glove_line(hand, fracs(tt - LAG_S, GLOVE), tt,
                                    counters["g"], tags))
        write_jsonl(base / "glove" / f"{name}.jsonl", grows)
        if camera:
            crows = []
            for i in range(int((t_end - t) * CAM_HZ)):
                tt = t + i / CAM_HZ
                counters["c"] += 1
                crows.append(camera_line(hand, fracs(tt, CAMERA), tt,
                                         counters["c"], tags, rng))
            write_jsonl(base / "leap" / f"{name}.jsonl", crows)
            if stills:
                (base / "stills").mkdir(parents=True, exist_ok=True)
                (base / "stills" / f"{name}.png").write_bytes(b"\x89PNG fake")
        events.append({"t": t_end, "kind": "take_end", "item": spec.item,
                       "take": n})
        events.append({"t": t_end + 0.5, "kind": "decision",
                       "accepted": spec.accepted, "reason": spec.reason,
                       "by": "operator" if not spec.accepted else "auto"})
        write_jsonl(base / "events" / f"{name}.events.jsonl", events)
        entries.append(_take_entry(session, name, spec.item, n, spec.accepted,
                                   spec.reason, t_end + 0.5,
                                   _files(name, camera, stills)))
        t = t_end + 3.0
    write_session_json(session, "sequences", hand, proto_path, camera,
                       entries)
    return session


GLOVE, CAMERA = pc.GLOVE, pc.CAMERA

GRASP_POSES = {"cylindrical": "fist", "tip_pinch": "thumb_opposition"}
GRASP_PROTOCOL = {"name": "grasps", "version": 1, "status": "placeholder",
                  "description": "test", "takes_per_item": 3,
                  "duration_s": 1.0, "prep_s": 0.0,
                  "items": [{"id": "cylindrical", "label": "cylindrical grasp",
                             "source": None, "figure": None,
                             "shape": "wrap", "object_implied": True},
                            {"id": "tip_pinch", "label": "tip pinch",
                             "source": None, "figure": None,
                             "shape": "pinch", "object_implied": False}]}


def glove_prof_exporter():
    import importlib.util
    path = REPO / "scripts" / "glove" / "export_prof_format.py"
    spec = importlib.util.spec_from_file_location("glove_prof_format_t", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_grasp_session(root: Path, hand="left", takes_per_item=3,
                       meta_overrides: Optional[Dict[str, dict]] = None,
                       rejected=True) -> Path:
    """A Set A session: camera only, both hands in view, meta + keypoints."""
    session = _session_dir(root, "grasps", hand)
    proto_path = write_protocol(root, GRASP_PROTOCOL)
    exporter = glove_prof_exporter()
    t = 1_790_200_000.0
    entries = []
    k = 0
    for item, pose in GRASP_POSES.items():
        attempts = [(n, True) for n in range(1, takes_per_item + 1)]
        if rejected and item == "tip_pinch":
            attempts.insert(1, (2, False))
        for n, accepted in attempts:
            name = f"{item}_{hand}_take{n}_{_stamp(k)}"
            k += 1
            base = session if accepted else session / "rejected"
            tags = {"session": session.name, "item": item, "take": n}
            stream = MockLeapStream(hz=90.0, pose=pose, seed=k,
                                    dropout_every=0, reacquire_every=0)
            stream._t0_us = int(t * 1e6)
            rows, own = [], []
            for side, lh in stream.generate(90):
                rows.append(leap_line(lh, lh.capture_time + 0.003, tags))
                if side == hand:
                    own.append(rows[-1])
            write_jsonl(base / "leap" / f"{name}.jsonl", rows)
            from xr_hand.recorder import _dict_to_frame
            frame, wall = _dict_to_frame(own[len(own) // 2])
            (base / "keypoints").mkdir(parents=True, exist_ok=True)
            (base / "keypoints" / f"{name}_keypoints.txt").write_text(
                exporter.frame_block(frame, wall) + "\n", encoding="utf-8")
            (base / "stills").mkdir(parents=True, exist_ok=True)
            (base / "stills" / f"{name}.png").write_bytes(b"\x89PNG fake")
            meta = {"item": item, "take": n, "hand": hand,
                    "frames": len(own), "tracked_fraction": 0.98,
                    "reacquisitions": 0,
                    "tracker_hand_labels": {"left": len(own),
                                            "right": len(rows) - len(own)},
                    "static_interval": [t + 0.1, t + 0.9],
                    "medoid_wall_time": wall,
                    "grab_strength": 0.9 if pose == "fist" else 0.2,
                    "pinch_strength": 0.1 if pose == "fist" else 0.8,
                    "curls": {f: 1.0 for f in FINGERS},
                    "orientation_note": "", "accepted": accepted,
                    "reason": "" if accepted else "fingertip wrong",
                    "decided_at": t + 2}
            meta.update((meta_overrides or {}).get(name.rsplit("_", 2)[0], {}))
            (base / "meta").mkdir(parents=True, exist_ok=True)
            (base / "meta" / f"{name}.json").write_text(
                json.dumps(meta, indent=1), encoding="utf-8")
            files = {"leap": f"leap/{name}.jsonl",
                     "still": f"stills/{name}.png",
                     "keypoints": f"keypoints/{name}_keypoints.txt",
                     "meta": f"meta/{name}.json"}
            entries.append(_take_entry(session, name, item, n, accepted,
                                       meta["reason"], t + 2, files))
            t += 10.0
    write_session_json(session, "grasps", hand, proto_path, True, entries)
    return session


def rows_by(check, **match):
    return [r for r in check.rows
            if all(r.get(k) == v for k, v in match.items())]


# --- shared sessions ---------------------------------------------------------------

@pytest.fixture(scope="module")
def flexion_session(tmp_path_factory):
    """index good, ring dragging middle and pinky, middle too small (rejected),
    then middle good."""
    root = tmp_path_factory.mktemp("flex")
    return make_flexion_session(root, [
        FlexTake("index"),
        FlexTake("ring", coupling={"middle": 0.8, "pinky": 0.45}),
        FlexTake("middle", peaks=[0.4] * 5, accepted=False,
                 reason="operator redo"),
        FlexTake("middle"),
    ])


@pytest.fixture(scope="module")
def flexion_check(flexion_session):
    return pc.check_session(flexion_session)


@pytest.fixture(scope="module")
def sequence_session(tmp_path_factory):
    """take 1 clean; take 2 glove reads the ring too open on middle + ring;
    take 3 the camera sees the index straight on index + middle."""
    root = tmp_path_factory.mktemp("seq")
    return make_sequence_session(root, [
        SeqTake(),
        SeqTake(glove_override={3: {"ring": 0.45}}),
        SeqTake(camera_override={2: {"index": 0.1}}),
    ])


@pytest.fixture(scope="module")
def sequence_check(sequence_session):
    return pc.check_session(sequence_session)


# --- Set B ---------------------------------------------------------------------------

def test_flexion_accepts_a_clean_take_and_measures_the_lag(flexion_check):
    (row,) = rows_by(flexion_check, item="index")
    assert row["verdict"] == pc.PASS, row["failures"]
    assert row["cued_finger"] == "index"
    assert row["cued_span_fraction"] > 0.9
    assert row["cycles_from_events"] == row["cycles_from_peaks"] == 5
    assert row["cycles_expected"] == 5
    assert row["cycles_camera_followed"] == 5
    assert row["frames_camera_followed"] > 0
    # the glove trails by LAG_S: per cycle and over the take, never one
    # number for the hand
    assert len(row["lag_ms_cycles"]) == 5
    assert abs(row["lag_ms_median"] - LAG_S * 1000) < 30
    assert abs(row["lag_ms_take"] - LAG_S * 1000) < 30
    assert row["paired_fraction"] > 0.95
    assert row["clock"] == "capture_time"
    assert 25 < row["glove_hz"] < 35
    assert row["glove_gap_max_ms"] < 100


def test_other_finger_coupling_is_reported_not_failed(flexion_check):
    (row,) = rows_by(flexion_check, item="ring")
    assert row["verdict"] == pc.PASS, row["failures"]
    assert row["other_spans"]["middle"] > 0.7
    assert 0.35 < row["other_spans"]["pinky"] < 0.55
    assert row["other_spans"]["index"] < 0.05
    assert "ring" not in row["other_spans"]


def test_rejected_attempt_is_checked_but_does_not_fail_the_session(
        flexion_check):
    rejected, accepted = (rows_by(flexion_check, item="middle", accepted=False),
                          rows_by(flexion_check, item="middle", accepted=True))
    assert len(rejected) == 1 and len(accepted) == 1
    assert rejected[0]["verdict"] == pc.FAIL
    assert rejected[0]["reason"] == "operator redo"
    assert any("spans 0.4" in f for f in rejected[0]["failures"])
    assert accepted[0]["verdict"] == pc.PASS
    assert flexion_check.exit_code == 0


def test_flexion_rejects_a_span_below_the_gate(tmp_path):
    session = make_flexion_session(tmp_path, [FlexTake("index",
                                                       peaks=[0.45] * 5)])
    check = pc.check_session(session)
    (row,) = check.rows
    assert row["verdict"] == pc.FAIL
    assert 0.35 < row["cued_span_fraction"] < 0.55
    assert any("need 0.60" in f for f in row["failures"])
    # its bends are still counted: the span rule is what failed
    assert row["cycles_from_peaks"] == 5
    assert check.exit_code == 1


def test_flexion_flags_a_missed_cycle(tmp_path):
    session = make_flexion_session(tmp_path, [FlexTake(
        "index", peaks=[1.0, 1.0, 0.0, 1.0, 1.0])])
    (row,) = pc.check_session(session).rows
    assert row["cycles_from_events"] == 5
    assert row["cycles_from_peaks"] == 4
    assert row["verdict"] == pc.FAIL
    assert any("5 bend cue(s) but 4 glove curl peak(s)" in f
               for f in row["failures"])


def test_flexion_checks_the_protocol_repetition_count(tmp_path):
    session = make_flexion_session(tmp_path, [FlexTake("index", cycles=4)])
    (row,) = pc.check_session(session).rows
    assert row["cycles_from_events"] == row["cycles_from_peaks"] == 4
    assert any("the protocol asks for 5" in f for f in row["failures"])


def test_camera_that_did_not_follow_gets_no_curve_or_lag(tmp_path):
    session = make_flexion_session(tmp_path, [FlexTake(
        "ring", camera_follows=False)])
    check = pc.check_session(session)
    (row,) = check.rows
    assert row["verdict"] == pc.PASS        # the glove is judged, not the camera
    assert row["camera_span"] < pc.MIN_CAMERA_RANGE
    assert row["cycles_camera_followed"] == 0
    assert row["lag_ms_median"] is None and row["lag_ms_take"] is None
    text = pc.render_text(check)
    assert "camera did not follow the ring" in text


def test_tracker_label_is_not_trusted(tmp_path):
    """The operator's left hand, every frame labelled 'right' by the tracker:
    the frames are still the take's camera (chosen by count, paired with the
    glove), but the palm normal is computed with the tracker's own chirality,
    as `fuse_poses.py` does, so this mirrored fit reads as facing away from
    the lens and the view gate keeps it out of the reference."""
    session = make_flexion_session(tmp_path, [FlexTake(
        "index", camera_label="right")])
    (row,) = pc.check_session(session).rows
    assert row["camera_label"] == "right"
    assert row["frames_camera"] > 0
    assert row["paired_fraction"] > 0.95
    assert any("labelled the hand 'right'" in n for n in row["notes"])
    assert row["camera_trusted_fraction"] == 0.0
    assert row["cycles_camera_followed"] == 0
    assert row["verdict"] == pc.PASS          # the glove still carries the take


def test_count_peaks_ignores_noise_on_a_hold():
    t = np.linspace(0, 10, 1000)
    trace = 0.5 - 0.5 * np.cos(2 * np.pi * t / 2.0)       # 5 bends
    noisy = trace + np.random.default_rng(0).normal(0, 0.03, t.size)
    assert pc.count_peaks(noisy) == 5
    assert pc.count_peaks(np.full(100, 0.5)) == 0
    # a take that starts bent does not count that first bend
    assert pc.count_peaks(np.concatenate([[1.0] * 20, noisy])) == 5


def test_camera_label_choice():
    assert pc.choose_camera_label({"left": 10, "right": 10}, "left")[0] == \
        "left"
    label, note = pc.choose_camera_label({"left": 3, "right": 500}, "left")
    assert label == "right" and "operator's hand is left" in note
    assert pc.choose_camera_label({}, "left") == (None, "")


# --- Set C ---------------------------------------------------------------------------

def test_sequence_clean_take_passes_every_step(sequence_check):
    row = rows_by(sequence_check, take=1)[0]
    assert row["verdict"] == pc.PASS, row["failures"]
    assert row["steps_total"] == 7
    assert row["steps_pass_glove"] == 7
    assert row["steps_judged_camera"] == 7
    assert row["steps_pass_camera"] == 7
    all_step = row["steps"][5]
    assert all_step["flexed"] == ALL
    assert all(v > 0.9 for v in all_step["glove"].values())


def test_glove_fault_fails_the_take_while_the_camera_passes(sequence_check):
    row = rows_by(sequence_check, take=2)[0]
    assert row["verdict"] == pc.FAIL
    assert row["steps_pass_glove"] == 6
    assert row["steps_pass_camera"] == 7
    step = row["steps"][3]
    assert step["glove_pass"] is False and step["camera_pass"] is True
    assert 0.35 < step["glove"]["ring"] < 0.55
    assert any("ring" in w and "not flexed" in w for w in step["glove_wrong"])


def test_camera_disagreement_is_reported_not_failed(sequence_check):
    row = rows_by(sequence_check, take=3)[0]
    assert row["verdict"] == pc.PASS, row["failures"]
    assert row["steps_pass_glove"] == 7
    assert row["steps_pass_camera"] == 6
    step = row["steps"][2]
    assert step["glove_pass"] is True and step["camera_pass"] is False
    assert sequence_check.exit_code == 1          # take 2 is accepted and fails


def test_bands_auto_rederives_from_the_session(tmp_path):
    """The glove reads 55 % of its warm-up range all session: the fixed bands
    fail every flexed finger, the session's own open and fist steps do not."""
    session = make_sequence_session(tmp_path, [SeqTake(glove_scale=0.55),
                                               SeqTake(glove_scale=0.55)],
                                    camera=False)
    fixed = pc.check_session(session, bands=pc.FIXED)
    auto = pc.check_session(session, bands=pc.AUTO)
    assert fixed.exit_code == 1
    assert all(r["verdict"] == pc.FAIL for r in fixed.rows)
    assert auto.exit_code == 0, [r["failures"] for r in auto.rows]
    assert all(r["steps_pass_glove"] == 7 for r in auto.rows)
    assert any("open from 4 open-hand step(s), fist from 2 full-fist step(s)"
               in h for h in auto.header)
    # the continuous value is on the session's scale: a flexed finger ~1.0
    assert auto.rows[0]["steps"][5]["glove"]["index"] > 0.9
    assert fixed.rows[0]["steps"][5]["glove"]["index"] < 0.6


@pytest.fixture(scope="module")
def coupling_session(tmp_path_factory):
    """Step 1 (thumb + index): the middle, which should be straight, rides
    along to 0.45 in take 1 and to 0.65 in take 2, on the glove only."""
    return make_sequence_session(tmp_path_factory.mktemp("cpl"), [
        SeqTake(glove_override={1: {"middle": 0.45}}),
        SeqTake(glove_override={1: {"middle": 0.65}})])


@pytest.mark.parametrize("bands", [pc.FIXED, pc.AUTO])
def test_straight_finger_between_the_bands_is_coupling_not_failure(
        coupling_session, bands):
    check = pc.check_session(coupling_session, bands=bands)
    coupled, flexed = check.rows
    # 0.45: a straight finger between 0.3 and 0.6 passes with a note
    assert coupled["verdict"] == pc.PASS, coupled["failures"]
    step = coupled["steps"][1]
    assert step["glove_pass"] is True
    assert list(step["coupling"][pc.GLOVE]) == ["middle"]
    assert 0.4 < step["coupling"][pc.GLOVE]["middle"] < 0.5
    assert step["glove"]["middle"] == step["coupling"][pc.GLOVE]["middle"]
    assert step["coupling"][pc.CAMERA] == {}
    assert coupled["coupling_glove"] == 1 and coupled["coupling_camera"] == 0
    # 0.65: a straight finger that reads flexed fails the step and the take
    assert flexed["verdict"] == pc.FAIL
    step = flexed["steps"][1]
    assert step["glove_pass"] is False and step["camera_pass"] is True
    assert any("middle 0.65 reads flexed" in w for w in step["glove_wrong"])
    text = pc.render_text(check)
    assert "glove coupling (not a failure): middle 0.4" in text
    assert "coupling g/c" in text


def test_step_rule_boundaries():
    """fraction = 1 - curl with these endpoints, so each curl is a known
    fraction: flexed needs above 0.6, straight fails at 0.6 or more, 0.3 to
    0.6 is coupling, below 0.3 is clean."""
    ref = {f: (1.0, 0.0) for f in FINGERS}

    def judged(fracs, flexed):
        curl = {f: 1.0 - fracs.get(f, 0.0) for f in FINGERS}
        return pc.judge_step(curl, flexed, ref)

    ok, _fr, wrong, cpl = judged({"index": 0.61, "middle": 0.59,
                                  "ring": 0.30, "pinky": 0.29}, ["index"])
    assert ok and wrong == [] and cpl == {"middle": 0.59, "ring": 0.3}
    ok, _fr, wrong, _c = judged({"index": 0.60}, ["index"])
    assert not ok and wrong == ["index 0.60 not flexed"]
    ok, _fr, wrong, _c = judged({"index": 0.9, "middle": 0.60}, ["index"])
    assert not ok and wrong == ["middle 0.60 reads flexed, should be "
                                "straight"]
    assert pc.judge_step(None, ["index"], ref) == (
        None, {f: None for f in FINGERS}, [], {})


def test_sequence_without_a_warmup(tmp_path):
    """No warmup.json: the fixed bands cannot be applied and say so once;
    the auto bands still can, from the open-hand and full-fist steps; the
    camera, with no endpoints either way under fixed, is not judged rather
    than failed."""
    session = make_sequence_session(tmp_path, [SeqTake()])
    (session / "warmup.json").unlink()
    fixed = pc.check_session(session, bands=pc.FIXED)
    (row,) = fixed.rows
    assert row["verdict"] == pc.FAIL
    assert row["failures"] == ["no glove endpoints to put the steps on a "
                               "fraction (no warm-up; try --bands auto)"]
    assert row["steps_judged_camera"] == 0 and row["steps_pass_glove"] == 0
    (row,) = pc.check_session(session, bands=pc.AUTO).rows
    assert row["verdict"] == pc.PASS, row["failures"]
    assert row["steps_pass_glove"] == 7 and row["steps_pass_camera"] == 7


def test_csv_values_keep_their_digits():
    assert pc.csv_value(1790200000.123456) == "1790200000.123456"
    assert pc.csv_value(0.98) == "0.98" and pc.csv_value(98.0) == "98"
    assert pc.csv_value(-0.0) == "0" and pc.csv_value(-1e-9) == "0"
    assert pc.csv_value(None) == "" and pc.csv_value(float("nan")) == ""
    assert pc.csv_value(True) == "true"
    assert pc.fmt(-0.001) == "0.00" and pc.fmt(0.0, "+.3f") == "0.000"
    assert pc.fmt(-0.25) == "-0.25"


# --- glove only, outputs, exit codes -----------------------------------------------

def test_glove_only_sessions_run(tmp_path):
    flex = make_flexion_session(tmp_path / "b", [FlexTake("index")],
                                camera=False, stills=False)
    seq = make_sequence_session(tmp_path / "c", [SeqTake()], camera=False,
                                stills=False)
    assert not (flex / "leap").exists() and not (flex / "stills").exists()
    for session in (flex, seq):
        check = pc.check_session(session)
        (row,) = check.rows
        assert row["verdict"] == pc.PASS, row["failures"]
        assert row["frames_camera"] is None
        assert row["paired_fraction"] is None
        assert check.exit_code == 0
        pc.write_check(check)
        assert (session / "check.csv").is_file()
    (row,) = pc.check_session(seq).rows
    assert row["steps_judged_camera"] == 0 and row["steps_pass_glove"] == 7


def test_check_csv_has_the_contract_columns(flexion_session, tmp_path):
    check = pc.check_session(flexion_session)
    csv_path, txt_path, text = pc.write_check(check, tmp_path / "out")
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    contract = ("set, item, take, accepted, reason, frames_glove, glove_hz, "
                "glove_gap_max_ms, frames_camera, paired_fraction, "
                "cued_span_fraction, other_spans, cycles_from_events, "
                "cycles_from_peaks, steps_total, steps_pass_glove, "
                "steps_pass_camera, lag_ms_median, verdict").split(", ")
    assert [c for c in contract if c not in header] == []
    assert len(rows) == 4
    index = next(r for r in rows if r["item"] == "index")
    assert json.loads(index["other_spans"])["thumb"] < 0.05
    assert len(json.loads(index["lag_ms_cycles"])) == 5
    assert index["accepted"] == "true" and index["verdict"] == "pass"
    assert pc.check_is_current(pc.load_session(flexion_session), rows)
    # check.txt: the table, then the diagnostics per item, then the rules
    assert text.index("index_left_take1") < text.index("transfer curve")
    assert "Rules in force" in text
    assert "\u2014" not in text


def test_cli_exit_codes_and_out_folder(tmp_path, flexion_session, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_protocol_cli", REPO / "scripts" / "check_protocol.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main([str(flexion_session), "--out", str(tmp_path / "a")]) == 0
    assert (tmp_path / "a" / "check.csv").is_file()
    assert (tmp_path / "a" / "check.txt").is_file()
    bad = make_flexion_session(tmp_path / "bad", [FlexTake("index",
                                                           peaks=[0.3] * 5)])
    assert cli.main([str(bad)]) == 1
    assert (bad / "check.csv").is_file()
    assert "REDO: index_left_take1" in capsys.readouterr().out
    assert cli.main([str(tmp_path / "nowhere")]) == 2


def test_changed_protocol_file_is_not_used(tmp_path):
    session = make_flexion_session(tmp_path, [FlexTake("index", cycles=4)])
    proto = tmp_path / "protocols" / "finger_flexion.json"
    proto.write_text(proto.read_text(encoding="utf-8") + "\n",
                     encoding="utf-8")
    check = pc.check_session(session)
    assert "sha256 differs" in check.header[0]
    (row,) = check.rows
    assert row["cycles_expected"] is None
    assert row["verdict"] == pc.PASS          # 4 cues, 4 peaks, no protocol


def test_missing_events_fails_the_take(tmp_path):
    session = make_flexion_session(tmp_path, [FlexTake("index")])
    for p in (session / "events").iterdir():
        p.unlink()
    (row,) = pc.check_session(session).rows
    assert row["verdict"] == pc.FAIL
    assert "no bend cues in the events" in row["failures"]
    assert "no events file" in row["notes"]


# --- Set A ---------------------------------------------------------------------------

def test_grasps_light_pass(tmp_path):
    session = make_grasp_session(tmp_path, meta_overrides={
        "cylindrical_left_take2": {"tracked_fraction": 0.80},
        "cylindrical_left_take3": {"grab_strength": 0.2}})
    check = pc.check_session(session)
    assert len(check.rows) == 7                      # 6 accepted, 1 rejected
    low = rows_by(check, name=next(r["name"] for r in check.rows
                                   if r["name"].startswith(
                                       "cylindrical_left_take2")))[0]
    assert low["verdict"] == pc.FAIL
    assert any("need 0.90" in f for f in low["failures"])
    odd = next(r for r in check.rows
               if r["name"].startswith("cylindrical_left_take3"))
    assert odd["verdict"] == pc.PASS
    assert any("second look: grab_strength" in n for n in odd["notes"])
    assert check.exit_code == 1
    assert all(r["frames_camera"] == 90 for r in check.rows)


def test_grasp_gate_counts_reacquisitions_inside_the_static_interval(
        tmp_path):
    """The Set A recorder counts the whole take in `reacquisitions` and the
    static interval under `gate`; only the interval can fail the gate."""
    session = make_grasp_session(tmp_path, rejected=False, meta_overrides={
        "cylindrical_left_take1": {
            "reacquisitions": 2,
            "gate": {"reacquisitions_in_static_interval": 0}},
        "cylindrical_left_take2": {
            "reacquisitions": 1,
            "gate": {"reacquisitions_in_static_interval": 1}},
        "cylindrical_left_take3": {"reacquisitions": 1}})
    rows = {r["take"]: r for r in pc.check_session(session).rows
            if r["item"] == "cylindrical"}
    assert rows[1]["verdict"] == pc.PASS, rows[1]["failures"]
    assert rows[1]["reacquisitions"] == 2
    assert "2 re-acquisition(s) outside the static interval" in rows[1][
        "notes"]
    assert rows[2]["verdict"] == pc.FAIL
    # no gate block: the contract's own field is all there is
    assert rows[3]["verdict"] == pc.FAIL


def test_a_summary_frame_fitted_as_the_other_hand_fails_the_grasp(tmp_path):
    """2026-10-01 13:40:28: the recorder measured the idle right hand of a
    left session. The meta names the tracker's label on the summary frame,
    and a label that is not the operator's hand fails the take; another
    hand in view beside the operator's is only a note."""
    session = make_grasp_session(tmp_path, rejected=False, meta_overrides={
        "cylindrical_left_take1": {"operator_hand_label": "right",
                                   "other_hand_ids": [25]},
        "cylindrical_left_take2": {"operator_hand_label": "left",
                                   "operator_hand_ids": [25],
                                   "other_hand_ids": [24]},
        "cylindrical_left_take3": {"operator_hand_label": None}})
    rows = {r["take"]: r for r in pc.check_session(session).rows
            if r["item"] == "cylindrical"}
    assert rows[1]["verdict"] == pc.FAIL
    assert ("summary frame is the tracker's right hand (operator's hand left)"
            in rows[1]["failures"])
    assert rows[2]["verdict"] == pc.PASS, rows[2]["failures"]
    assert "another hand in view (id 24), ignored" in rows[2]["notes"]
    assert not any("another hand" in n for n in rows[3]["notes"])
    assert rows[3]["verdict"] == pc.PASS, rows[3]["failures"]


def test_step_window_keeps_its_share_of_a_scaled_hold():
    from types import SimpleNamespace
    t = 1000.0
    data = SimpleNamespace(events=[
        {"t": t, "kind": "cue", "step": 0, "label": "open hand",
         "flexed": [], "hold_s": 2.5},
        {"t": t + 2.5, "kind": "cue", "step": 1, "label": "little flexion",
         "flexed": ["little"], "hold_s": 1.0}])
    first, second = pc.step_windows(data, 2.5, 1.5)
    assert first["window"] == pytest.approx([t + 1.0, t + 2.5])
    assert second["window"] == pytest.approx([t + 2.5 + 0.4, t + 3.5])
    assert second["flexed"] == ["pinky"]


def test_mock_flags_in_all_three_shapes(tmp_path):
    assert pc.mock_flags({"mock": True}) == ["mock"]
    assert pc.mock_flags({"mock": {"glove": True, "leap": False}}) == [
        "mock.glove"]
    assert pc.mock_flags({"mock": False, "mock_flags": {"leap": True}}) == [
        "mock_flags.leap"]
    assert pc.mock_flags({"mock": {"glove": False}, "mock_flags": {}}) == []
    assert pc.mock_flags({}) == []
    session = make_flexion_session(tmp_path, [FlexTake("index")],
                                   camera=False, stills=False)
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    meta["mock"] = {"glove": True, "glove_follows_cues": True, "leap": False}
    (session / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    text = pc.render_text(pc.check_session(session))
    assert "MOCK      synthetic data" in text and "mock.glove" in text
