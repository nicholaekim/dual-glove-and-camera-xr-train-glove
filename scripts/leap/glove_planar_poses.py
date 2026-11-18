"""50 single-plane static poses to record with the XR glove, not yet in the database.

The glove senses finger flexion (MCP and PIP from its sensors, DIP from its
hand model) and a coupled thumb; it has no sensor for finger spread and none
for the wrist. So every pose here moves each digit only in its own
flexion-extension plane: spread "together", wrist "neutral", thumb in a state
that stays in the thumb's own bending plane, and no hyperextension state (the
glove's export never reads below 0 degrees of flexion). That is the "one
plane" of a colleague's test: record new poses and see whether the measured angles
satisfy the constraint tables compiled on 7 Nov 2026.

Digit-state templates, the literature envelope and the recorded-database
exclusion list are imported from scripts/leap/pose_catalogue.py (the camera
catalogue of the same day), so both lists share one set of numbers. A pose is
excluded when its FINGER SHAPE (the four finger states, whatever the thumb,
spread or wrist) is the finger shape of a recorded pose (ten paired poses,
July seven, Set C steps, mimed grasps) or of one of the 100 camera-catalogue
poses: the glove does not sense spread or the wrist and its thumb is its
weakest estimate, so a pose that differs from a Leap pose only there would be
the same pose to the glove. No two glove poses share a finger shape either.

    .venv\\Scripts\\python.exe scripts\\leap\\glove_planar_poses.py

writes protocols/glove_planar_poses_50.csv, protocols/glove_planar_poses_50.json
(a recorder pose list) and docs/glove_planar_poses_50.md.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "protocols" / "glove_planar_poses_50.csv"
JSON_PATH = REPO / "protocols" / "glove_planar_poses_50.json"
DOC_PATH = REPO / "docs" / "glove_planar_poses_50.md"


def _catalogue():
    spec = importlib.util.spec_from_file_location(
        "pose_catalogue", Path(__file__).resolve().parent / "pose_catalogue.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("pose_catalogue", mod)
    spec.loader.exec_module(mod)
    return mod


pc = _catalogue()
Pose = pc.Pose
FINGERS, FINGER_NAME = pc.FINGERS, pc.FINGER_NAME

# Glove only (the user's correction of 13 Nov 2026): the Leap poses are one
# set and these are a separate, different set, recorded without the camera.
SETUP = ("gloves on, no camera: XR Train open and streaming, the hand held still "
         "in the pose for the 5 s of the take")
PLANE = ("each digit moves only in its own flexion-extension plane: fingers side by side "
         "(spread together), wrist neutral, thumb bent or straight in its own plane")
PLANAR_THUMB = ("up", "along", "across", "tucked", "ip_bent", "mcp_bent")
GLOVE_FLOOR_STATES = ("hyper",)       # the glove never reports hyperextension
# The names carry no count: the size of the recording set has changed twice
# (50, then 25, then 10), and only RECORDING_SET below should have to change.
LAUNCHER = "launchers\\record_glove_poses.cmd"
RECORDING_NAME = "glove_planar_poses_recording_set"
RECORD_COMMAND = (".venv\\Scripts\\python.exe scripts\\glove\\record_poses.py "
                  f"--hand left --pose-list protocols\\{RECORDING_NAME}.json --takes 3 "
                  "--first-take auto --duration 5 --prep 5 --hz 0 "
                  "--out-dir recordings\\glove_planar_left --app-screenshot \"XR Train\"")
JSON_REC_PATH = REPO / "protocols" / f"{RECORDING_NAME}.json"
# The poses of the 50 that are recorded: 10 since 13 Nov 2026 ("ok lets only
# do 10 poses for the glove"; 25 earlier that day). One pose for each kind of
# test, so every group, every finger state and the three thumb-joint tests
# are still in it:
#   ps01 one finger, PIP and DIP alone      pd01 MCP alone at 90 (two fingers)
#   pf07 PIP and DIP at their limit (hook)  pr07 a quarter curl beside a full curl
#   pc01, pc05 one finger full, its neighbour held half (index-middle, ring-little)
#   pm01 graded depths across the hand      pt01, pt02, pt03 the thumb's tip
#   joint alone, its MCP alone, and the thumb across the palm
# In catalogue order.
RECORDING_SET = (
    "ps01",
    "pd01",
    "pf07",
    "pr07",
    "pc01", "pc05",
    "pm01",
    "pt01", "pt02", "pt03",
)
RECORDING_REASON = {
    "ps01": "one finger's PIP and DIP bent alone, MCP straight",
    "pd01": "the MCP joint alone at its 90 degree limit",
    "pf07": "PIP and DIP at their limit with the MCP straight (hook)",
    "pr07": "a quarter curl beside a full curl",
    "pc01": "index fully curled with the middle held half: coupling between neighbours",
    "pc05": "ring fully curled with the little held half: coupling on the ulnar side",
    "pm01": "graded depths across the hand, quarter and half",
    "pt01": "the thumb's tip joint alone",
    "pt02": "the thumb's MCP joint alone",
    "pt03": "the thumb across the palm",
}

# What each digit state asks of the constraint tables (the glove's view).
TESTS = {
    "tabletop": "MCP flexion limit with PIP and DIP at zero (MCP 90 in Isaac, Chen and Jia; 80 in Dragusanu)",
    "hook": "PIP and DIP flexion with the MCP straight (PIP 100 to 135 across the sources, MyoHand stops at 90) and the PIP ~ 3/2 DIP relation",
    "tipcurl": "mid-range PIP and DIP with the MCP straight: the PIP ~ 3/2 DIP relation (Paper 3 eq. 8) without a full curl",
    "quarter": "graded flexion: a quarter curl should sit well inside every table",
    "half": "graded flexion: a half curl should sit inside every table, MCP and PIP moving together",
    "flex": "full flexion against MCP 90 and PIP 110 to 135 (the pilot glove export topped out at 90 to 100)",
    "ext": "a straight finger: MCP, PIP and DIP near zero while another digit moves",
}
THUMB_TESTS = {
    "up": "thumb straight in the palm plane: thumb MCP and IP near zero",
    "along": "thumb adducted along the index: thumb MCP and IP near zero, TMC adducted",
    "across": "thumb half flexed across the palm: thumb MCP 20 to 50 and IP 20 to 60 against the MCP limits 45 (MyoHand), 50 (Jia), 75 to 80 (Chen, Isaac)",
    "tucked": "thumb fully flexed into the palm: thumb MCP and IP at their flexion limits (IP 75 MyoHand, 80 Jia, 90 Isaac)",
    "ip_bent": "thumb IP flexion alone (40 to 80) with the thumb MCP straight",
    "mcp_bent": "thumb MCP flexion alone (30 to 55) with the thumb IP straight",
}
COUPLING = ("inter-finger coupling (Paper 3 eq. 3 to 7: the ring follows the middle, the little "
            "follows the ring) when one finger is held at a different depth from its neighbour")


def thumb_words(state: str) -> str:
    """The thumb state for the glove: the catalogue's words, with 'abducted' read as
    'in the thumb's own plane' because the glove does not sense the TMC's abduction."""
    return {
        "up": "straight, in the palm plane, pointing along the radial side",
        "along": "adducted, lying along the index metacarpal",
        "across": "half flexed across the palm, over the fingers",
        "tucked": "fully flexed into the palm, under or against the fingers",
        "ip_bent": "straight at the MCP, only the tip joint bent",
        "mcp_bent": "bent at the MCP only, tip joint straight",
    }[state]


# --- the 50 poses ----------------------------------------------------------------------------
def poses() -> list:
    P = []
    E, F, H, Q, K, T, C = "ext", "flex", "half", "quarter", "hook", "tabletop", "tipcurl"

    def p(pid, name, group, desc, flex, thumb="up"):
        P.append(Pose(pid, name, group, desc, tuple(flex), thumb, "", "together", "neutral"))

    # S. one finger with PIP and DIP half bent, the other three straight
    S = "single finger"
    p("ps01", "middle tip curl, others straight", S, "only the middle finger with PIP and DIP half bent and its MCP straight; the other three straight and together, thumb straight in the palm plane", (E, C, E, E))
    p("ps02", "ring tip curl, others straight", S, "only the ring finger with PIP and DIP half bent and its MCP straight; the other three straight and together, thumb straight in the palm plane", (E, E, C, E))

    # D. two fingers in one state, the other two straight
    D = "pair"
    p("pd01", "middle and ring tabletop", D, "middle and ring bent 90 at the MCP only, index and little straight, thumb up in plane", (E, T, T, E))
    p("pd02", "middle and ring half curled", D, "middle and ring half curled, index and little straight, thumb up in plane", (E, H, H, E))
    p("pd03", "middle and ring tip curl", D, "middle and ring with PIP and DIP half bent and MCP straight, index and little straight", (E, C, C, E))
    p("pd04", "index and ring tabletop", D, "index and ring bent 90 at the MCP only, middle and little straight", (T, E, T, E))
    p("pd05", "index and little tabletop", D, "index and little bent 90 at the MCP only, middle and ring straight", (T, E, E, T))
    p("pd06", "middle and little tabletop", D, "middle and little bent 90 at the MCP only, index and ring straight", (E, T, E, T))
    p("pd07", "index and middle tip curl, others straight", D, "index and middle with PIP and DIP half bent, ring and little straight", (C, C, E, E))
    p("pd08", "ring and little tip curl", D, "ring and little with PIP and DIP half bent, index and middle straight", (E, E, C, C))
    p("pd09", "index and ring half curled", D, "index and ring half curled, middle and little straight", (H, E, H, E))
    p("pd10", "middle and little half curled", D, "middle and little half curled, index and ring straight", (E, H, E, H))
    p("pd11", "middle and ring quarter curled", D, "middle and ring a quarter curled, index and little straight", (E, Q, Q, E))
    p("pd12", "index and little quarter curled", D, "index and little a quarter curled, middle and ring straight", (Q, E, E, Q))

    # F. three fingers in one state, the fourth straight
    Fg = "three fingers"
    p("pf01", "index straight, three tabletop", Fg, "middle, ring and little bent 90 at the MCP with PIP and DIP straight, index straight", (E, T, T, T))
    p("pf02", "little straight, three tabletop", Fg, "index, middle and ring bent 90 at the MCP with PIP and DIP straight, little straight", (T, T, T, E))
    p("pf03", "index straight, three tip curl", Fg, "middle, ring and little with PIP and DIP half bent and MCP straight, index straight", (E, C, C, C))
    p("pf04", "little straight, three tip curl", Fg, "index, middle and ring with PIP and DIP half bent and MCP straight, little straight", (C, C, C, E))
    p("pf05", "index straight, three quarter curled", Fg, "middle, ring and little a quarter curled, index straight", (E, Q, Q, Q))
    p("pf06", "little straight, three quarter curled", Fg, "index, middle and ring a quarter curled, little straight", (Q, Q, Q, E))
    p("pf07", "little straight, three hooked", Fg, "index, middle and ring bent at PIP and DIP with MCP straight, little straight", (K, K, K, E))

    # R. radial pair (index, middle) and ulnar pair (ring, little) in two different states
    R = "split"
    p("pr01", "ring and little quarter curled", R, "ring and little a quarter curled, index and middle straight", (E, E, Q, Q))
    p("pr02", "radial tabletop, ulnar quarter", R, "index and middle bent 90 at the MCP only, ring and little a quarter curled", (T, T, Q, Q))
    p("pr03", "radial quarter, ulnar tabletop", R, "index and middle a quarter curled, ring and little bent 90 at the MCP only", (Q, Q, T, T))
    p("pr04", "radial half, ulnar tabletop", R, "index and middle half curled, ring and little bent 90 at the MCP only", (H, H, T, T))
    p("pr05", "radial hook, ulnar half", R, "index and middle hooked at PIP and DIP, ring and little half curled", (K, K, H, H))
    p("pr06", "radial half, ulnar hook", R, "index and middle half curled, ring and little hooked at PIP and DIP", (H, H, K, K))
    p("pr07", "radial quarter, ulnar full", R, "index and middle a quarter curled, ring and little fully curled", (Q, Q, F, F))
    p("pr08", "radial full, ulnar quarter", R, "index and middle fully curled, ring and little a quarter curled", (F, F, Q, Q))
    p("pr09", "radial tabletop, ulnar full", R, "index and middle bent 90 at the MCP only, ring and little fully curled", (T, T, F, F))

    # C. coupling: one finger fully curled, its neighbour held at a set depth
    Cg = "coupling"
    p("pc01", "index full, middle half", Cg, "index fully curled, middle held half curled, ring and little straight", (F, H, E, E))
    p("pc02", "middle full, index half", Cg, "middle fully curled, index held half curled, ring and little straight", (H, F, E, E))
    p("pc03", "middle full, ring half", Cg, "middle fully curled, ring held half curled, index and little straight", (E, F, H, E))
    p("pc04", "ring full, middle half", Cg, "ring fully curled, middle held half curled, index and little straight", (E, H, F, E))
    p("pc05", "ring full, little half", Cg, "ring fully curled, little held half curled, index and middle straight", (E, E, F, H))
    p("pc06", "little full, ring half", Cg, "little fully curled, ring held half curled, index and middle straight", (E, E, H, F))
    p("pc07", "index full, middle quarter", Cg, "index fully curled, middle held a quarter curled, ring and little straight", (F, Q, E, E))
    p("pc08", "little full, ring quarter", Cg, "little fully curled, ring held a quarter curled, index and middle straight", (E, E, Q, F))

    # M. mixed depths across the hand
    M = "mixed depth"
    p("pm01", "radial quarter, ulnar half", M, "index and middle a quarter curled, ring and little half curled", (Q, Q, H, H))
    p("pm02", "radial half, ulnar quarter", M, "index and middle half curled, ring and little a quarter curled", (H, H, Q, Q))
    p("pm03", "inner fingers half, outer quarter", M, "middle and ring half curled, index and little a quarter curled", (Q, H, H, Q))
    p("pm04", "outer fingers half, inner quarter", M, "index and little half curled, middle and ring a quarter curled", (H, Q, Q, H))
    p("pm05", "radial tabletop, ulnar half", M, "index and middle bent 90 at the MCP only, ring and little half curled", (T, T, H, H))
    p("pm06", "radial hook, ulnar full", M, "index and middle hooked at PIP and DIP, ring and little fully curled", (K, K, F, F))

    # T. one thumb joint at a time, each on a finger shape used nowhere else
    Tg = "thumb"
    p("pt01", "radial tip curl, ulnar half, thumb tip bent", Tg, "index and middle with PIP and DIP half bent, ring and little half curled, thumb straight at its MCP with only the tip joint bent", (C, C, H, H), "ip_bent")
    p("pt02", "radial half, ulnar tip curl, thumb MCP bent", Tg, "index and middle half curled, ring and little with PIP and DIP half bent, thumb bent at its MCP only", (H, H, C, C), "mcp_bent")
    p("pt03", "radial full, ulnar tabletop, thumb across", Tg, "index and middle fully curled, ring and little bent 90 at the MCP only, thumb half flexed across the palm", (F, F, T, T), "across")
    p("pt04", "radial full, ulnar hook, thumb MCP bent", Tg, "index and middle fully curled, ring and little hooked at PIP and DIP, thumb bent at its MCP only", (F, F, K, K), "mcp_bent")
    p("pt05", "ring quarter, little half, thumb tip bent", Tg, "index and middle straight, ring a quarter curled, little half curled, only the thumb's tip joint bent", (E, E, Q, H), "ip_bent")
    p("pt06", "index half, middle quarter, thumb along", Tg, "index half curled, middle a quarter curled, ring and little straight, thumb adducted along the index", (H, Q, E, E), "along")
    return P


# --- exclusions ---------------------------------------------------------------------------------
CATALOGUE = pc.poses()
CATALOGUE_SIGNATURES = {q.signature for q in CATALOGUE}
EXCLUDED_SIGNATURES = set(pc.DATABASE_SIGNATURES) | CATALOGUE_SIGNATURES
# The stricter rule: the four finger states alone. A recorded or Leap pose with
# these fingers and any thumb, spread or wrist rules the finger shape out.
LEAP_FINGER_SHAPES = {q.flex for q in CATALOGUE}
DATABASE_FINGER_SHAPES = {tuple(sig.split("|")[:4]) for sig in pc.DATABASE_SIGNATURES}
EXCLUDED_FINGER_SHAPES = LEAP_FINGER_SHAPES | DATABASE_FINGER_SHAPES


def finger_shape(pose: Pose) -> str:
    return "|".join(pose.flex)


def is_planar(pose: Pose) -> bool:
    return (pose.abd == "together" and pose.wrist == "neutral"
            and pose.thumb in PLANAR_THUMB
            and not any(s in GLOVE_FLOOR_STATES for s in pose.flex))


def tests_text(pose: Pose) -> str:
    out = []
    for s in pose.flex:
        t = TESTS[s]
        if t not in out:
            out.append(t)
    if pose.category == "coupling":
        out.insert(0, COUPLING)
    out.append(THUMB_TESTS[pose.thumb])
    return "; ".join(out)


def glove_note(pose: Pose) -> str:
    notes = ["spread is not sensed: the glove reports a fixed splay, so abduction is not measured in this recording"]
    if any(s in ("hook", "tipcurl", "tabletop") for s in pose.flex):
        notes.append("the PIP-DIP split is the glove's hand model, not a separate DIP sensor")
    if "flex" in pose.flex:
        notes.append("the pilot export topped out at 90 (index, little PIP) and 100 (middle, ring PIP), so a full curl may read a ceiling")
    if pose.thumb != "up":
        notes.append("the thumb is a coupled three-component estimate whose TMC mapping is unverified")
    return "; ".join(notes)


# --- rows, json, note ------------------------------------------------------------------------------
def rows(plist) -> list:
    out = []
    for pose in plist:
        r = pc.ranges(pose)
        row = {
            "id": pose.id, "name": pose.name, "group": pose.category,
            "description": pose.description, "setup": SETUP, "plane": PLANE,
            "tests": tests_text(pose),
            "index_state": pose.flex[0], "middle_state": pose.flex[1],
            "ring_state": pose.flex[2], "little_state": pose.flex[3],
            "spread_state": pose.abd, "thumb_state": pose.thumb,
            "thumb_target": pose.thumb_target, "wrist_state": pose.wrist,
        }
        for col in pc.ANGLE_COLUMNS:
            row[f"{col}_min"], row[f"{col}_max"] = r[col]
        row["tmc_note"] = thumb_words(pose.thumb)
        row["glove_note"] = glove_note(pose)
        row["finger_shape"] = finger_shape(pose)
        row["recording_set"] = "yes" if pose.id in RECORDING_SET else "no"
        row["signature"] = pose.signature
        out.append(row)
    return out


def protocol(plist, name: str = "glove_planar_poses_50") -> dict:
    return {
        "name": name,
        "version": 1,
        "description": (f"Set D: {len(plist)} static single-plane poses for the XR glove, gloves on, "
                        "no camera, each held still; written by "
                        "scripts/leap/glove_planar_poses.py, ranges and exclusions in "
                        "protocols/glove_planar_poses_50.csv. Record with "
                        "scripts/glove/record_poses.py --pose-list this file."),
        "setup": SETUP,
        "plane": PLANE,
        "takes_per_item": 3,
        "duration_s": 5.0,
        "prep_s": 5.0,
        "items": [{"id": pose.id, "label": pose.name, "group": pose.category,
                   "shape": pose.description,
                   "orientation": "gloves on, no camera; move each digit only in its own bending plane, fingers side by side, wrist straight"}
                  for pose in plist],
    }


def _rng(r) -> str:
    return f"{r[0]} to {r[1]}"


def _finger_cell(pose: Pose, f: str, r: dict) -> str:
    state = pose.flex[FINGERS.index(f)]
    return f"{state}: MCP {_rng(r[f + '_MCP_fe'])}, PIP {_rng(r[f + '_PIP'])}, DIP {_rng(r[f + '_DIP'])}"


def document(plist) -> str:
    ids = ",".join(p.id for p in plist)
    L = []
    L.append("# 50 single-plane poses for the XR glove, with reference joint ranges\n")
    L.append("Written 7 Nov 2026 by `scripts/leap/glove_planar_poses.py`; the same data is "
             "`protocols/glove_planar_poses_50.csv`, and `protocols/glove_planar_poses_50.json` is the "
             "recorder's pose list. Gates: `GATES_glove_planar_poses.md`.\n")
    L.append("## What this is\n")
    L.append("Fifty static hand poses that are not in the database yet, chosen so the StretchSense XR "
             "glove can record them, each with an approximate reference range for every angle the "
             "glove reports: finger MCP flexion, PIP and DIP, thumb MCP and IP. The purpose is the test "
             "a colleague proposed: record new poses with the glove and see whether the measured angles satisfy "
             "the constraint tables compiled on 7 Nov 2026 (Dragusanu 2025 Table I, Isaac 2021 Table 1, "
             "Ingram 2008 Table 1, Chen 2013 Table 6 which is Cobos 2009 Table 5, Jia 2021 Table 1, and "
             "the MyoHand model). A range is what a held pose should look like, not a pass/fail limit. "
             "Each pose also says which constraint it exercises.\n")
    L.append("## Why one plane\n")
    L.append("The glove senses flexion: MCP and PIP from its sensors, DIP from its hand model, and the "
             "thumb as a coupled estimate. It has no sensor for finger spread (the pilot export shows a "
             "fixed splay) and none for the wrist, and its flexion never reads below 0, so it cannot show "
             "hyperextension. So every pose here keeps the fingers side by side, the wrist straight and "
             "the thumb in its own bending plane, and uses no hyperextension state: " + PLANE + ". A pose "
             "that needed spread, wrist motion or thumb opposition would be measured by the camera only, "
             "which is the camera catalogue's job (`docs/pose_catalogue_100.md`).\n")
    L.append("## How to read the ranges\n")
    L.append("Every pose is written as digit states and every state has one template range, imported "
             "from the camera catalogue so the two lists agree. Angles are the repo's "
             "(`docs/joint_frames.md`): flexion positive toward the palm. Spread is `together` and the "
             "wrist `neutral` throughout, so their columns are constant and are not repeated in the table "
             "(they are in the CSV). The thumb's TMC pair is not comparable across sources "
             "(`docs/tmc_estimate.md`), so the thumb carries its MCP and IP ranges and the TMC in words.\n")
    L.append("Finger flexion templates (MCP, PIP, DIP, degrees):\n")
    L.append("| state | MCP flexion | PIP | DIP | meaning |\n|---|---|---|---|---|")
    meaning = {"ext": "straight", "quarter": "a quarter curled", "half": "half curled",
               "flex": "fully curled", "hook": "MCP straight, PIP and DIP bent (claw)",
               "tabletop": "MCP bent 90, PIP and DIP straight", "tipcurl": "MCP straight, PIP and DIP half bent"}
    used = sorted({s for p in plist for s in p.flex}, key=list(pc.FLEX).index)
    for s in used:
        m, pp, d = pc.FLEX[s]
        L.append(f"| {s} | {_rng(m)} | {_rng(pp)} | {_rng(d)} | {meaning[s]} |")
    L.append("\nThumb states (thumb MCP flexion, thumb IP flexion; the TMC in words, for the glove):\n")
    L.append("| state | thumb MCP | thumb IP | TMC |\n|---|---|---|---|")
    for s in PLANAR_THUMB:
        m, i, _ = pc.THUMB[s]
        L.append(f"| {s} | {_rng(m)} | {_rng(i)} | {thumb_words(s)} |")
    L.append("\nEvery template lies inside the literature envelope of the camera catalogue "
             "(`docs/pose_catalogue_100.md`, table \"envelope\"): MCP flexion -40 to 90, PIP -5 to 135, "
             "DIP -30 to 90, thumb MCP -30 to 80, thumb IP -45 to 90. Two numbers matter for the glove in "
             "particular: MyoHand caps PIP at 90 where Chen allows 110 to 135, and the glove's own export "
             "topped out at 90 or 100 degrees of PIP in the pilot fist, so a full curl may read a ceiling "
             "rather than the finger.\n")
    L.append("## What is excluded as already in the database\n")
    L.append("The ten paired poses of 2026-11-05 (open palm, fist, index point, thumbs up, peace, pinch, "
             "finger spread, thumb opposition, crossed fingers, bent wrist), the July seven, the 23 mimed "
             "grasps of Set A, every step of the seven Set C sequences, and all 100 poses of the Leap "
             "(camera) catalogue of 7 Nov 2026. The exclusion is by **finger shape**: the four finger "
             "states alone, whatever the thumb, the spread or the wrist. A glove pose whose four fingers "
             "match a Leap pose or a recorded pose is left out even when its thumb or spread differs, "
             "because the glove does not sense spread or the wrist and its thumb is its weakest estimate, "
             "so to the glove it would be the same pose. "
             f"That rules out {len(EXCLUDED_FINGER_SHAPES)} finger shapes ({len(LEAP_FINGER_SHAPES)} from the Leap "
             f"list, {len(DATABASE_FINGER_SHAPES)} from the recorded poses, some shared), among them every "
             "whole-hand shape (all four fingers straight, quarter, half, tabletop, tip curl, hook or "
             "fist) and every single finger at tabletop, hook, half or quarter. No two of the 50 share a "
             "finger shape with each other either. `tests/test_glove_planar_poses.py` checks all of it. "
             "The professor's 102 tracker frames carry no pose names and are not matched.\n")
    L.append("## Recording setup\n")
    L.append(f"Every pose: {SETUP}. Setup for every step: gloves on, no camera. No objects. The glove "
             "recording uses no camera at all: the Leap poses are one set, recorded with the camera, and "
             "these are a separate, different set, recorded with the glove alone. "
             "One take is one held pose, three takes per pose, left hand. The "
             "glove note in the table says what the glove cannot sense in that pose. One launcher records "
             "the recording set with the glove-only recorder:\n")
    L.append("    " + LAUNCHER + "\n")
    L.append("It prints the setup, waits for a key, and runs:\n")
    L.append("    " + RECORD_COMMAND + "\n")
    rec_ids = ",".join(p.id for p in plist if p.id in RECORDING_SET)
    n_rec = len(RECORDING_SET)
    L.append(f"The recorder takes the {n_rec} poses of the recording set (below) from "
             f"`protocols/{RECORDING_NAME}.json` in this order (" + rec_ids + ") and, for "
             "each, prints the id with its name and description, counts down 5 s (beeps on 3, 2, 1) and "
             "records 5 s, keeping every frame of the left glove. To carry on after a "
             "stop, give the launcher the pose to start at, for example `" + LAUNCHER + " pc01`; takes "
             "are numbered after the ones already on disk.\n")
    L.append("**A screenshot of XR Train for every take.** XR Train must be open with its window not "
             "minimised (it may sit behind other windows). At the midpoint of every take's 5 s, "
             "the recorder saves a picture of the XR Train window to "
             "`recordings\\glove_planar_left\\xr_train\\<take>.png`, named like the take, and the "
             "take's `meta.json` says where it is, when and how it was captured "
             "(`app_screenshot*` keys), beside the pose, its name and description, the hand, the "
             "frame count and the take's start and end times. If "
             "a picture cannot be taken the take is still kept and its meta says why; the recorder "
             "refuses to start at all if the window cannot be captured, and the session's last lines "
             "count how many takes have a picture. Gates: `GATES_glove_recording.md`.\n")
    L.append("`protocols/glove_planar_poses_50.json` holds all 50 items in the same pose-list "
             "format (id, label, shape), for recording the others later with the same recorder.\n")
    L.append(f"## The {n_rec} to record\n")
    L.append(f"Asked for on 13 Nov 2026: record {n_rec} of the 50 (first 25, then 10). The {n_rec} "
             "are one pose for each kind of test, so every group, every finger "
             "state (tabletop, hook, tip curl, quarter, half, full curl) and the three thumb-joint "
             "tests (tip joint alone, MCP alone, across the palm) are still recorded; the table says "
             "what each one is there for. Left out are the variants of a kept pose on other fingers, "
             f"the mirror images, and the shapes that ask most of finger independence. The other "
             f"{len(plist) - n_rec} stay in the catalogue "
             "below, marked `-` in its `record` column, and can be recorded later with the same "
             f"recorder. `results/glove_planar_poses/{RECORDING_NAME}.pdf` shows only these {n_rec}.\n")
    L.append("| n | id | pose | group | why it is in the set |\n|---|---|---|---|---|")
    for n, pose in enumerate((p for p in plist if p.id in RECORDING_SET), 1):
        L.append(f"| {n} | `{pose.id}` | {pose.name} | {pose.category} | {RECORDING_REASON[pose.id]} |")
    L.append("")
    L.append("## The 50 poses\n")
    L.append("Each finger cell is `state: MCP, PIP, DIP` in degrees (spread together and wrist neutral "
             "throughout); the thumb cell is MCP, IP and the TMC in words; `tests` is the constraint the "
             "pose exercises.\n")
    L.append("| id | pose | index | middle | ring | little | thumb | tests | glove | record |\n|---|---|---|---|---|---|---|---|---|---|")
    for pose in plist:
        r = pc.ranges(pose)
        thumb = f"{pose.thumb}: MCP {_rng(r['T_MCP_fe'])}, IP {_rng(r['T_IP'])}; TMC {thumb_words(pose.thumb)}"
        L.append(f"| {pose.id} | **{pose.name}**: {pose.description} | "
                 + " | ".join(_finger_cell(pose, f, r) for f in FINGERS)
                 + f" | {thumb} | {tests_text(pose)} | {glove_note(pose)} | "
                 + ("yes" if pose.id in RECORDING_SET else "-") + " |")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--json", type=Path, default=JSON_PATH)
    ap.add_argument("--doc", type=Path, default=DOC_PATH)
    args = ap.parse_args(argv)
    plist = poses()
    problems = []
    if len(plist) != 50:
        problems.append(f"{len(plist)} poses, not 50")
    ids = [p.id for p in plist]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    sigs = [p.signature for p in plist]
    if len(set(sigs)) != len(sigs):
        problems.append("duplicate signatures: " + ", ".join(sorted({s for s in sigs if sigs.count(s) > 1})))
    taken = [f"{p.id}={p.signature}" for p in plist if p.signature in EXCLUDED_SIGNATURES]
    if taken:
        problems.append("already in the database or the camera catalogue: " + ", ".join(taken))
    shared = [f"{p.id}={finger_shape(p)}" for p in plist if p.flex in EXCLUDED_FINGER_SHAPES]
    if shared:
        problems.append("finger shape already in the Leap list or the database: " + ", ".join(shared))
    shapes = [p.flex for p in plist]
    if len(set(shapes)) != len(shapes):
        problems.append("two glove poses share a finger shape: "
                        + ", ".join(p.id for p in plist if shapes.count(p.flex) > 1))
    flat = [p.id for p in plist if not is_planar(p)]
    if flat:
        problems.append("not single-plane: " + ", ".join(flat))
    chosen = [p for p in plist if p.id in RECORDING_SET]
    if not RECORDING_SET or len(set(RECORDING_SET)) != len(RECORDING_SET) \
            or len(chosen) != len(RECORDING_SET):
        problems.append(f"the recording set names {len(RECORDING_SET)} poses, "
                        f"{len(chosen)} of them in the list, some twice or unknown")
    if set(RECORDING_REASON) != set(RECORDING_SET):
        problems.append("RECORDING_REASON does not give a reason for exactly the recording set")
    if [p.id for p in chosen] != list(RECORDING_SET):
        problems.append("the recording set is not in catalogue order")
    for what, have, want in (
            ("group", {p.category for p in chosen}, {p.category for p in plist}),
            ("finger state", {s for p in chosen for s in p.flex}, {s for p in plist for s in p.flex}),
            ("thumb test", {p.thumb for p in chosen}, {"up", "ip_bent", "mcp_bent", "across"})):
        if not want <= have:
            problems.append(f"the recording set has no {what} {', '.join(sorted(want - have))}")
    for pose in plist:
        for col, (lo, hi) in pc.ranges(pose).items():
            e = pc.ENVELOPE[pc.envelope_key(col)]
            if not (e[0] <= lo < hi <= e[1]):
                problems.append(f"{pose.id} {col} {lo}..{hi} outside {e}")
    if problems:
        print("GLOVE PLANAR POSES FAIL: " + "; ".join(problems))
        return 1
    table = rows(plist)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    args.json.write_text(json.dumps(protocol(plist), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    json_rec = args.json.with_name(JSON_REC_PATH.name)
    json_rec.write_text(json.dumps(protocol(chosen, RECORDING_NAME), indent=1,
                                   ensure_ascii=False) + "\n", encoding="utf-8")
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(document(plist), encoding="utf-8")
    groups = {}
    for pose in plist:
        groups[pose.category] = groups.get(pose.category, 0) + 1
    print(f"wrote {args.csv}")
    print(f"wrote {args.json}")
    print(f"wrote {args.doc}")
    print(f"GLOVE PLANAR POSES OK: poses={len(plist)} groups="
          + ",".join(f"{k}={v}" for k, v in groups.items()) + " shared_finger_shapes_with_leap_or_database=0"
          + f" recording_set={len(chosen)} ({json_rec.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
