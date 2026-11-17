"""100 static hand poses for the Leap Motion, with reference joint ranges.

Each pose is written as digit states (one flexion state per finger, one
abduction pattern, one thumb state, one wrist state); the ranges come from
one template table per state, so every pose's numbers are consistent and
every number sits inside the envelope of the constraint tables compiled on
7 Nov 2026 (Dragusanu 2025 Table I, Isaac 2021 Table 1, Ingram 2008 Table 1,
Chen 2013 Table 6 = Cobos 2009 Table 5, Jia 2021 Table 1, MyoHand). The
ranges are reference ranges for a held pose, not limits: a recording that
lands inside them looks like the pose; one outside is worth a look, not a
fail.

Angles are the camera's (xr_hand.joint_frames): flexion positive toward the
palm, abduction positive toward the thumb side, wrist flexion positive toward
the palm, wrist deviation positive toward the thumb (radial). The thumb's TMC
pair is not comparable across sources (docs/tmc_estimate.md), so the thumb
carries its MCP and IP ranges and a description of the TMC in words.

Database poses are excluded by digit-state signature: the ten paired poses,
the July seven, the Set C sequence steps and the mimed grasp shapes.

    .venv\\Scripts\\python.exe scripts\\leap\\pose_catalogue.py

writes protocols/pose_catalogue_100.csv and docs/pose_catalogue_100.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "protocols" / "pose_catalogue_100.csv"
DOC_PATH = REPO / "docs" / "pose_catalogue_100.md"

SETUP = "camera only, bare hand, palm toward the camera about 30 cm above it, held still 3 s"

# --- templates: (min, max) degrees -------------------------------------------------------
# Finger flexion states -> (MCP flexion, PIP, DIP)
FLEX = {
    "ext":      ((-10, 15), (0, 10), (0, 10)),      # straight
    "hyper":    ((-35, -10), (-5, 5), (-10, 5)),    # pressed back (Isaac -40, Chen 30-40 hyper)
    "quarter":  ((15, 35), (15, 35), (5, 20)),      # slightly curled
    "half":     ((35, 60), (35, 65), (15, 45)),     # half curled
    "flex":     ((70, 90), (85, 110), (50, 90)),    # fully curled (PIP from Chen 110, Isaac 130)
    "hook":     ((0, 20), (70, 100), (40, 80)),     # MCP straight, PIP and DIP bent (claw)
    "tabletop": ((70, 90), (0, 15), (0, 10)),       # MCP bent, PIP and DIP straight
    "tipcurl":  ((0, 15), (30, 60), (30, 60)),      # MCP straight, PIP and DIP half bent
}
# Abduction patterns -> per finger (I, M, R, L) MCP abduction, positive toward the thumb
ABD = {
    "together": {"I": (-5, 5), "M": (-5, 5), "R": (-5, 5), "L": (-5, 5)},
    "neutral":  {"I": (0, 10), "M": (-5, 5), "R": (-5, 5), "L": (-10, 0)},
    "spread":   {"I": (10, 25), "M": (0, 10), "R": (-10, 0), "L": (-25, -10)},
    "vulcan":   {"I": (5, 20), "M": (5, 15), "R": (-15, -5), "L": (-20, -5)},
    "scissors": {"I": (15, 25), "M": (-10, 0), "R": (-5, 5), "L": (-10, 0)},
    "ulnar_split": {"I": (0, 10), "M": (-5, 5), "R": (0, 10), "L": (-25, -10)},
}
# Thumb states -> (thumb MCP flexion, thumb IP flexion, TMC in words)
THUMB = {
    "out":      ((-10, 15), (-10, 15), "abducted and extended, away from the palm"),
    "up":       ((-10, 15), (-10, 15), "extended in the palm plane, pointing along the radial side"),
    "hyper_up": ((-30, -10), (-30, -10), "extended in the palm plane and pressed back (hitch-hiker)"),
    "along":    ((0, 20), (0, 20), "adducted, lying along the index metacarpal"),
    "across":   ((20, 50), (20, 60), "flexed across the palm, over the fingers"),
    "tucked":   ((30, 55), (40, 80), "flexed into the palm, under the fingers"),
    "between":  ((20, 45), (10, 40), "flexed, tip between the index and middle fingers"),
    "oppose":   ((15, 45), (5, 40), "opposed, tip touching the named fingertip"),
    "side":     ((20, 50), (10, 50), "adducted, tip touching the named finger's base or side"),
    "c_curve":  ((5, 30), (0, 25), "abducted and slightly flexed, curved opposite the fingers, no contact"),
    "ip_bent":  ((-10, 15), (40, 80), "abducted and extended, only the tip joint bent"),
    "mcp_bent": ((30, 55), (-10, 10), "abducted, bent at the MCP only, tip joint straight"),
}
# Wrist states -> (wrist flexion, wrist deviation)
WRIST = {
    "neutral":  ((-15, 15), (-10, 10)),
    "flexed":   ((40, 70), (-10, 10)),
    "extended": ((-70, -40), (-10, 10)),
    "radial":   ((-15, 15), (10, 20)),
    "ulnar":    ((-15, 15), (-35, -20)),
}
# Envelope every range must lie inside: the widest literature values compiled on 7 Nov 2026.
ENVELOPE = {
    "MCP_fe": (-40, 90),    # Isaac -40 hyperextension, 90 flexion (Chen, Jia, Isaac)
    "MCP_aa": (-30, 30),    # Chen 60 total, Jia 0-60
    "PIP": (-5, 135),       # Chen little PIP 135, Isaac 130
    "DIP": (-30, 90),       # Isaac -30 to 90
    "T_MCP_fe": (-30, 80),  # Ingram -22 observed, Isaac/Dragusanu 80
    "T_IP": (-45, 90),      # Ingram -43 observed, Isaac 90
    "WRIST_fe": (-80, 80),  # Jia flex 80 / ext 70, Dragusanu +-90 model
    "WRIST_dev": (-35, 35), # Dragusanu +-35, Jia ulnar 30
}

FINGERS = ("I", "M", "R", "L")
FINGER_NAME = {"I": "index", "M": "middle", "R": "ring", "L": "little"}


@dataclass(frozen=True)
class Pose:
    id: str
    name: str
    category: str
    description: str
    flex: tuple      # (I, M, R, L) flexion states
    thumb: str
    thumb_target: str = ""
    abd: str = "neutral"
    wrist: str = "neutral"

    @property
    def signature(self) -> str:
        return "|".join(list(self.flex) + [self.abd, self.thumb, self.thumb_target, self.wrist])

    def leap_note(self) -> str:
        if self.thumb in ("tucked", "between") or (self.thumb == "across" and all(f == "flex" for f in self.flex)):
            return "fair: the thumb is hidden under or behind the fingers, the Leap infers it"
        if any(f in ("hook", "tipcurl") for f in self.flex):
            return "fair: curled fingertips can hide behind the proximal phalanx; keep the palm square to the camera"
        if any(f == "hyper" for f in self.flex) or self.thumb == "hyper_up":
            return "good, but the Leap rarely reports more than about 20 degrees of hyperextension"
        if self.wrist != "neutral":
            return "good: keep the forearm in view so the wrist frame is stable"
        return "good: palm toward the camera, every fingertip visible"


# --- database signatures to exclude ----------------------------------------------------------
def _db() -> set:
    """Digit-state signatures of poses already in the database.

    Ten paired poses (2026-11-05), the July seven, the Set C sequence steps
    (any thumb state, neutral wrist), and the mimed grasp shapes of Set A.
    crossed_fingers has no digit-state signature and is not representable here.
    """
    out = set()

    def add(flex, thumbs, abds=("together", "neutral", "spread"), targets=("",), wrists=("neutral",)):
        for t in thumbs:
            for a in abds:
                for tg in targets:
                    for w in wrists:
                        out.add("|".join(list(flex) + [a, t, tg, w]))

    E, F, H = "ext", "flex", "half"
    all_thumbs = tuple(THUMB)
    # open_palm, finger_spread (and the July open palm): flat hand, thumb free
    add((E, E, E, E), ("out", "up"), abds=("neutral", "spread"))
    # bent_wrist: open hand, wrist flexed
    add((E, E, E, E), all_thumbs, wrists=("flexed",))
    # fist (July fist, ASL S/M/N shapes), thumbs_up, lateral/key grasp
    add((F, F, F, F), ("across", "tucked", "up", "side"), targets=("", "index base", "index side"))
    # index_point (and ASL 1/D/G shapes with the thumb folded)
    add((E, F, F, F), ("across", "tucked", "along"))
    # peace / V (ASL U/H shapes)
    add((E, E, F, F), ("across", "tucked", "along"))
    # July three (ASL W)
    add((E, E, E, F), ("across", "tucked", "along"))
    # pinch, tip grasp, OK/ASL F/9: thumb to the index tip
    for i_state in ("quarter", H, "tipcurl", "hook"):
        for others in ((E, E, E), (F, F, F), (H, H, H)):
            add((i_state,) + others, ("oppose", "side"), targets=("index", "index tip"))
    # thumb_opposition (thumb to the little fingertip), ASL 6
    for l_state in ("quarter", H, "tipcurl", "hook"):
        for others in ((E, E, E), (F, F, F)):
            add(others + (l_state,), ("oppose",), targets=("little",))
    # Set C sequence steps: single fingers, adjacent pairs, progressive fists, thumb alone
    straight = ("out", "up", "along")
    bent = ("across", "tucked")
    for flex in [(F, E, E, E), (E, F, E, E), (E, E, F, E), (E, E, E, F),   # one finger
                 (F, F, E, E), (E, F, F, E), (E, E, F, F)]:                # adjacent pairs
        add(flex, straight)
    for flex in [(F, E, E, E), (F, F, E, E), (F, F, F, E)]:                # thumb + fingers
        add(flex, bent)
    add((E, E, E, E), ("across", "tucked"))          # "thumb flexion" step: fingers straight
    # mimed grasps (Set A): power/cylindrical wraps, hook grasp, palmar/spherical/prismatic
    # precision, tripod and the circular precision family
    add((H, H, H, H), ("across", "tucked", "oppose", "side"), targets=("", "all fingertips", "index"))
    add(("hook", "hook", "hook", "hook"), all_thumbs)
    add((H, H, F, F), ("oppose",), targets=("", "index", "middle", "index and middle"))
    add((H, H, H, F), ("oppose",), targets=("", "index", "middle"))
    return out


DATABASE_SIGNATURES = _db()


# --- the 100 poses ---------------------------------------------------------------------------
def poses() -> list:
    P = []
    E, F, H, Q, K, T, C, Y = "ext", "flex", "half", "quarter", "hook", "tabletop", "tipcurl", "hyper"

    def p(pid, name, cat, desc, flex, thumb, target="", abd="neutral", wrist="neutral"):
        P.append(Pose(pid, name, cat, desc, tuple(flex), thumb, target, abd, wrist))

    # A. fingerspelling and number handshapes (static ASL shapes not in the database)
    A = "ASL handshape"
    p("A01", "ASL A", A, "fist with the thumb straight alongside the index, not across the fingers", (F, F, F, F), "along", abd="together")
    p("A03", "ASL C", A, "fingers together and slightly curled, thumb curved opposite them, an open C", (Q, Q, Q, Q), "c_curve", abd="together")
    p("A04", "ASL D", A, "index straight up, middle, ring and little half curled with their tips on the thumb", (E, H, H, H), "oppose", "middle")
    p("A07", "ASL K", A, "index up, middle slightly curled, thumb tip between them on the middle finger's base, ring and little curled", (E, Q, F, F), "side", "middle base")
    p("A08", "ASL L", A, "index up and thumb out, an L; middle, ring and little curled", (E, F, F, F), "out")
    p("A09", "ASL T", A, "fist with the thumb tip pushed between the index and middle fingers", (F, F, F, F), "between", abd="together")
    p("A10", "ASL X", A, "index bent to a hook, other fingers curled, thumb across them", (K, F, F, F), "across")
    p("A11", "ASL Y (shaka)", A, "thumb and little finger out, index, middle and ring curled", (F, F, F, E), "out")
    p("A14", "ASL 7", A, "thumb tip on the ring fingertip, index, middle and little straight", (E, E, H, E), "oppose", "ring")
    p("A15", "ASL 8", A, "thumb tip on the middle fingertip, index, ring and little straight", (E, H, E, E), "oppose", "middle")

    # B. one finger at a time, in states Set B and Set C never record
    B = "single finger"
    for f, i in zip(FINGERS, range(4)):
        n = FINGER_NAME[f]
        def one(state, rest=E):
            return tuple(state if j == i else rest for j in range(4))
        p(f"B{i*5+1:02d}", f"{n} half curled, others straight", B, f"only the {n} finger half curled, thumb out", one(H), "out")
        p(f"B{i*5+2:02d}", f"{n} hooked, others straight", B, f"only the {n} finger bent at PIP and DIP with its MCP straight, thumb out", one(K), "out")
        p(f"B{i*5+3:02d}", f"{n} tabletop, others straight", B, f"only the {n} finger bent 90 at the MCP with PIP and DIP straight, thumb out", one(T), "out")
        p(f"B{i*5+4:02d}", f"{n} hyperextended, others straight", B, f"only the {n} finger pressed back past straight, the others flat, thumb out", one(Y), "out", abd="spread")
        p(f"B{i*5+5:02d}", f"{n} slightly curled, others straight", B, f"only the {n} finger curled a quarter of the way, thumb out", one(Q), "out")
    p("B21", "middle finger up, others curled", B, "middle straight, index, ring and little curled, thumb across", (F, E, F, F), "across")
    p("B22", "ring finger up, others curled", B, "ring straight, index, middle and little curled, thumb across", (F, F, E, F), "across")

    # C. whole-hand graded flexion with the thumb out of the way (no object mimed)
    Cg = "whole hand"
    p("C01", "quarter curl, thumb out", Cg, "all four fingers curled a quarter of the way, together, thumb out", (Q, Q, Q, Q), "out", abd="together")
    p("C02", "half fist, thumb out", Cg, "all four fingers half curled, thumb out of the palm plane", (H, H, H, H), "out", abd="together")
    p("C03", "tabletop, thumb out", Cg, "all four fingers bent 90 at the MCP, PIP and DIP straight, thumb out", (T, T, T, T), "out", abd="together")
    p("C04", "tabletop, thumb along", Cg, "tabletop fingers with the thumb adducted along the index", (T, T, T, T), "along", abd="together")
    p("C05", "tip curl, thumb out", Cg, "MCPs straight, PIP and DIP of every finger half bent, thumb out", (C, C, C, C), "out", abd="together")
    p("C06", "fist, thumb out of plane", Cg, "fingers fully curled, thumb abducted straight out of the palm plane", (F, F, F, F), "out", abd="together")
    p("C07", "open claw, fingers apart", Cg, "fingers half curled and spread apart, thumb out", (H, H, H, H), "out", abd="spread")
    p("C08", "fingers pressed back", Cg, "all four fingers hyperextended as on a flat surface, thumb out", (Y, Y, Y, Y), "out", abd="spread")
    p("C09", "flat hand, fingers together, thumb out", Cg, "flat hand with the four fingers touching, thumb out", (E, E, E, E), "out", abd="together")
    p("C11", "half fist, thumb up", Cg, "all four fingers half curled, thumb straight up in the palm plane, a loose thumbs-up", (H, H, H, H), "up", abd="together")
    p("C10", "relaxed open hand", Cg, "fingers a quarter curled and slightly apart, thumb out, a resting hand", (Q, Q, Q, Q), "out", abd="spread")

    # D. thumb placements with the fingers flat
    D = "thumb"
    p("D01", "flat hand, thumb along", D, "flat hand, fingers together, thumb adducted along the index", (E, E, E, E), "along", abd="together")
    p("D02", "spread fingers, thumb along", D, "fingers spread, thumb adducted along the index", (E, E, E, E), "along", abd="spread")
    p("D04", "thumb to middle base", D, "fingers flat and spread, thumb tip on the base of the middle finger", (E, E, E, E), "side", "middle base", abd="spread")
    p("D05", "thumb to index base", D, "fingers flat and spread, thumb tip on the base of the index finger", (E, E, E, E), "side", "index base", abd="spread")
    p("D06", "thumb to ring base", D, "fingers flat and spread, thumb tip on the base of the ring finger", (E, E, E, E), "side", "ring base", abd="spread")
    p("D07", "thumb to little base", D, "fingers flat and spread, thumb tip on the base of the little finger", (E, E, E, E), "side", "little base", abd="spread")
    p("D08", "thumb tip bent", D, "fingers spread and flat, thumb out with only its tip joint bent", (E, E, E, E), "ip_bent", abd="spread")
    p("D09", "thumb MCP bent", D, "fingers spread and flat, thumb out and bent at its MCP with the tip straight", (E, E, E, E), "mcp_bent", abd="spread")
    p("D13", "quarter curl, thumb tucked", D, "fingers a quarter curled and together, thumb folded into the palm under them", (Q, Q, Q, Q), "tucked", abd="together")
    p("D10", "hitch-hiker", D, "fingers curled, thumb pressed back past straight", (F, F, F, F), "hyper_up", abd="together")
    p("D11", "thumb to middle tip, others curled", D, "thumb tip on the middle fingertip, index, ring and little curled", (F, H, F, F), "oppose", "middle")
    p("D12", "thumb to ring tip, others curled", D, "thumb tip on the ring fingertip, index, middle and little curled", (F, F, H, F), "oppose", "ring")

    # E. finger combinations not among the sequence steps
    Ec = "combination"
    p("E01", "index and ring curled", Ec, "index and ring fully curled, middle and little straight, thumb out", (F, E, F, E), "out")
    p("E02", "index and little curled", Ec, "index and little fully curled, middle and ring straight, thumb out", (F, E, E, F), "out")
    p("E03", "middle and little curled", Ec, "middle and little fully curled, index and ring straight, thumb out", (E, F, E, F), "out")
    p("E04", "index and middle half curled", Ec, "index and middle half curled, ring and little straight, thumb out", (H, H, E, E), "out")
    p("E05", "ring and little half curled", Ec, "ring and little half curled, index and middle straight, thumb out", (E, E, H, H), "out")
    p("E06", "index and middle hooked", Ec, "index and middle bent at PIP and DIP only, ring and little straight, thumb out", (K, K, E, E), "out")
    p("E07", "ring and little hooked", Ec, "ring and little bent at PIP and DIP only, index and middle straight, thumb out", (E, E, K, K), "out")
    p("E08", "Vulcan salute", Ec, "all fingers straight, index with middle and ring with little, a gap between middle and ring, thumb out", (E, E, E, E), "out", abd="vulcan")
    p("E09", "index-middle scissors", Ec, "all fingers straight, index and middle spread wide, ring and little together", (E, E, E, E), "out", abd="scissors")
    p("E10", "little finger split", Ec, "all fingers straight, little finger abducted away, the other three together", (E, E, E, E), "out", abd="ulnar_split")
    p("E11", "rock horns", Ec, "index and little straight, middle and ring curled under the thumb", (E, F, F, E), "across")
    p("E17", "index and middle tip curl", Ec, "index and middle with PIP and DIP half bent, ring and little curled, thumb out", (C, C, F, F), "out")
    p("E18", "index and middle tabletop", Ec, "index and middle bent 90 at the MCP only, ring and little straight, thumb out", (T, T, E, E), "out")
    p("E19", "ring and little tabletop", Ec, "ring and little bent 90 at the MCP only, index and middle straight, thumb out", (E, E, T, T), "out")
    p("E21", "middle and ring hooked", Ec, "middle and ring bent at PIP and DIP only, index and little straight, thumb out", (E, K, K, E), "out")
    p("E20", "index and middle quarter curl", Ec, "index and middle a quarter curled, ring and little straight, thumb out", (Q, Q, E, E), "out")

    # F. staircases and mixed depths
    Fm = "mixed depth"
    p("F01", "staircase down", Fm, "index straight, middle a quarter, ring half, little fully curled, thumb out", (E, Q, H, F), "out")
    p("F02", "staircase up", Fm, "index fully curled, middle half, ring a quarter, little straight, thumb out", (F, H, Q, E), "out")
    p("F03", "outer fingers half", Fm, "index and little half curled, middle and ring fully curled, thumb out", (H, F, F, H), "out")
    p("F04", "inner fingers half", Fm, "middle and ring half curled, index and little fully curled, thumb out", (F, H, H, F), "out")
    p("F05", "radial half, ulnar full", Fm, "index and middle half curled, ring and little fully curled, thumb out", (H, H, F, F), "out")
    p("F06", "radial full, ulnar half", Fm, "index and middle fully curled, ring and little half curled, thumb out", (F, F, H, H), "out")
    p("F07", "claw with index pointing", Fm, "index straight, middle, ring and little hooked at PIP and DIP, thumb out", (E, K, K, K), "out")
    p("F08", "tabletop, thumb across", Fm, "all four fingers bent 90 at the MCP only, thumb across the palm", (T, T, T, T), "across", abd="together")
    p("F09", "half fist, thumb along", Fm, "all four fingers half curled, thumb adducted along the index", (H, H, H, H), "along", abd="together")
    p("F11", "index up, rest half", Fm, "index straight, middle, ring and little half curled, thumb out", (E, H, H, H), "out")
    p("F12", "little up, rest half", Fm, "little straight, index, middle and ring half curled, thumb out", (H, H, H, E), "out")

    # G. wrist positions (the database has only wrist flexion with an open hand)
    G = "wrist"
    p("G01", "open hand, wrist extended", G, "fingers spread and straight, thumb out, wrist bent back", (E, E, E, E), "out", abd="spread", wrist="extended")
    p("G02", "open hand, radial deviation", G, "fingers spread and straight, thumb out, hand tilted toward the thumb side", (E, E, E, E), "out", abd="spread", wrist="radial")
    p("G03", "open hand, ulnar deviation", G, "fingers spread and straight, thumb out, hand tilted toward the little-finger side", (E, E, E, E), "out", abd="spread", wrist="ulnar")
    p("G04", "fist, wrist flexed", G, "fist with the thumb across, wrist bent toward the palm", (F, F, F, F), "across", abd="together", wrist="flexed")
    p("G05", "fist, wrist extended", G, "fist with the thumb across, wrist bent back", (F, F, F, F), "across", abd="together", wrist="extended")
    p("G06", "fist, radial deviation", G, "fist with the thumb across, hand tilted toward the thumb side", (F, F, F, F), "across", abd="together", wrist="radial")
    p("G07", "fist, ulnar deviation", G, "fist with the thumb across, hand tilted toward the little-finger side", (F, F, F, F), "across", abd="together", wrist="ulnar")
    p("G08", "half fist, wrist flexed", G, "fingers half curled, thumb out, wrist bent toward the palm", (H, H, H, H), "out", abd="together", wrist="flexed")
    p("G09", "half fist, wrist extended", G, "fingers half curled, thumb out, wrist bent back", (H, H, H, H), "out", abd="together", wrist="extended")
    p("G10", "index point, wrist extended", G, "index straight, others curled under the thumb, wrist bent back", (E, F, F, F), "across", wrist="extended")
    p("G11", "index point, wrist flexed", G, "index straight, others curled under the thumb, wrist bent toward the palm", (E, F, F, F), "across", wrist="flexed")
    p("G12", "flat hand together, wrist extended", G, "flat hand, fingers together, thumb along, wrist bent back", (E, E, E, E), "along", abd="together", wrist="extended")
    p("G13", "L shape, wrist extended", G, "index up and thumb out, other fingers curled, wrist bent back", (E, F, F, F), "out", wrist="extended")
    p("G16", "tabletop, wrist extended", G, "fingers bent 90 at the MCP only, thumb out, wrist bent back", (T, T, T, T), "out", abd="together", wrist="extended")
    p("G17", "open claw, wrist flexed", G, "fingers half curled and spread, thumb out, wrist bent toward the palm", (H, H, H, H), "out", abd="spread", wrist="flexed")
    p("G18", "tip curl, wrist extended", G, "MCPs straight with PIP and DIP half bent, thumb out, wrist bent back", (C, C, C, C), "out", abd="together", wrist="extended")
    p("G19", "shaka, ulnar deviation", G, "thumb and little finger out, other fingers curled, hand tilted toward the little-finger side", (F, F, F, E), "out", wrist="ulnar")
    p("G21", "ASL A, wrist flexed", G, "fist with the thumb along the index, wrist bent toward the palm", (F, F, F, F), "along", abd="together", wrist="flexed")
    return P


# --- ranges ------------------------------------------------------------------------------------
ANGLE_COLUMNS = (
    [f"{f}_{a}" for f in FINGERS for a in ("MCP_fe", "MCP_aa", "PIP", "DIP")]
    + ["T_MCP_fe", "T_IP", "WRIST_fe", "WRIST_dev"])


def ranges(pose: Pose) -> dict:
    out = {}
    for f, state in zip(FINGERS, pose.flex):
        mcp, pip, dip = FLEX[state]
        out[f"{f}_MCP_fe"] = mcp
        out[f"{f}_MCP_aa"] = ABD[pose.abd][f]
        out[f"{f}_PIP"] = pip
        out[f"{f}_DIP"] = dip
    tm, ti, _ = THUMB[pose.thumb]
    out["T_MCP_fe"], out["T_IP"] = tm, ti
    out["WRIST_fe"], out["WRIST_dev"] = WRIST[pose.wrist]
    return out


def envelope_key(col: str) -> str:
    for k in ("MCP_fe", "MCP_aa", "PIP", "DIP"):
        if col.endswith(k) and not col.startswith(("T_", "WRIST")):
            return k
    return col


def rows(plist) -> list:
    out = []
    for pose in plist:
        r = ranges(pose)
        row = {
            "id": pose.id, "name": pose.name, "category": pose.category,
            "description": pose.description, "setup": SETUP,
            "index_state": pose.flex[0], "middle_state": pose.flex[1],
            "ring_state": pose.flex[2], "little_state": pose.flex[3],
            "spread_state": pose.abd, "thumb_state": pose.thumb,
            "thumb_target": pose.thumb_target, "wrist_state": pose.wrist,
        }
        for col in ANGLE_COLUMNS:
            row[f"{col}_min"], row[f"{col}_max"] = r[col]
        row["tmc_note"] = THUMB[pose.thumb][2] + (f" ({pose.thumb_target})" if pose.thumb_target else "")
        row["leap_note"] = pose.leap_note()
        row["signature"] = pose.signature
        out.append(row)
    return out


# --- the note ----------------------------------------------------------------------------------
def _rng(r) -> str:
    return f"{r[0]} to {r[1]}"


def _finger_cell(pose: Pose, f: str, r: dict) -> str:
    state = pose.flex[FINGERS.index(f)]
    return (f"{state}: MCP {_rng(r[f + '_MCP_fe'])}, PIP {_rng(r[f + '_PIP'])}, "
            f"DIP {_rng(r[f + '_DIP'])}, abd {_rng(r[f + '_MCP_aa'])}")


def document(plist) -> str:
    L = []
    L.append("# 100 new poses for the Leap Motion, with reference joint ranges\n")
    L.append("Written 7 Nov 2026 by `scripts/leap/pose_catalogue.py`; the same data is "
             "`protocols/pose_catalogue_100.csv`. Gates: `GATES_pose_catalogue.md`.\n")
    L.append("## What this is\n")
    L.append("One hundred static hand poses that are not in the database yet, chosen so the "
             "Ultraleap IR 170 can see them (camera only, bare hand), each with an approximate "
             "reference range for every angle the camera gives: finger MCP flexion and abduction, "
             "PIP, DIP, thumb MCP and IP, wrist flexion and deviation. The purpose is the test a colleague "
             "proposed: record new poses and see whether the measured angles satisfy the constraint "
             "tables. A range here is what a held pose should look like, not a pass/fail limit: a "
             "recording inside it looks like the pose, one outside it is worth a look.\n")
    L.append("## How to read the ranges\n")
    L.append("Every pose is written as digit states, and every state has one template range, so the "
             "numbers are consistent across the catalogue. Angles are the camera's "
             "(`docs/joint_frames.md`): flexion positive toward the palm, abduction positive toward "
             "the thumb side, wrist flexion positive toward the palm, wrist deviation positive toward "
             "the thumb (radial). The thumb's TMC pair is not comparable across sources "
             "(`docs/tmc_estimate.md`), so the thumb carries its MCP and IP ranges and the TMC in words.\n")
    L.append("Finger flexion templates (MCP, PIP, DIP, degrees):\n")
    L.append("| state | MCP flexion | PIP | DIP | meaning |\n|---|---|---|---|---|")
    meaning = {"ext": "straight", "hyper": "pressed back past straight", "quarter": "a quarter curled",
               "half": "half curled", "flex": "fully curled", "hook": "MCP straight, PIP and DIP bent (claw)",
               "tabletop": "MCP bent 90, PIP and DIP straight", "tipcurl": "MCP straight, PIP and DIP half bent"}
    for s, (m, pp, d) in FLEX.items():
        L.append(f"| {s} | {_rng(m)} | {_rng(pp)} | {_rng(d)} | {meaning[s]} |")
    L.append("\nAbduction patterns (MCP abduction per finger, positive toward the thumb):\n")
    L.append("| pattern | index | middle | ring | little |\n|---|---|---|---|---|")
    for s, d in ABD.items():
        L.append(f"| {s} | {_rng(d['I'])} | {_rng(d['M'])} | {_rng(d['R'])} | {_rng(d['L'])} |")
    L.append("\nThumb states (thumb MCP flexion, thumb IP flexion; the TMC in words):\n")
    L.append("| state | thumb MCP | thumb IP | TMC |\n|---|---|---|---|")
    for s, (m, i, w) in THUMB.items():
        L.append(f"| {s} | {_rng(m)} | {_rng(i)} | {w} |")
    L.append("\nWrist states (flexion, deviation):\n")
    L.append("| state | wrist flexion | wrist deviation |\n|---|---|---|")
    for s, (fe, dev) in WRIST.items():
        L.append(f"| {s} | {_rng(fe)} | {_rng(dev)} |")
    L.append("\nEvery template lies inside this envelope, the widest values of the constraint tables "
             "compiled on 7 Nov 2026 (Dragusanu 2025 Table I, Isaac 2021 Table 1, Ingram 2008 Table 1, "
             "Chen 2013 Table 6 which is Cobos 2009 Table 5, Jia 2021 Table 1, and the MyoHand model):\n")
    L.append("| angle | envelope | where the bound comes from |\n|---|---|---|")
    src = {"MCP_fe": "Isaac -40 hyperextension; 90 flexion in Isaac, Chen and Jia",
           "MCP_aa": "Chen 60 total, Jia 0 to 60, so plus or minus 30",
           "PIP": "Chen little PIP 135, Isaac 130; MyoHand stops at 90 and is the tightest",
           "DIP": "Isaac -30 to 90",
           "T_MCP_fe": "Ingram observed -22; Isaac and Dragusanu 80",
           "T_IP": "Ingram observed -43; Isaac 90",
           "WRIST_fe": "Jia flexion 80, extension 70; Dragusanu plus or minus 90 is a model limit",
           "WRIST_dev": "Dragusanu plus or minus 35; Jia ulnar 30, radial 20"}
    for k, v in ENVELOPE.items():
        L.append(f"| {k} | {_rng(v)} | {src[k]} |")
    L.append("\nThe templates are deliberately narrower than the envelope: a full curl is written as "
             "PIP 85 to 110 although the literature allows 135, because a held fist in the pilot takes "
             "read 90 to 100, and a straight finger is written as MCP -10 to 15 because the Leap rarely "
             "reports more hyperextension than that on a relaxed hand.\n")
    L.append("## What is excluded as already in the database\n")
    L.append("The ten paired poses of 2026-11-05 (open palm, fist, index point, thumbs up, peace, "
             "pinch, finger spread, thumb opposition, crossed fingers, bent wrist), the July seven "
             "(the same plus the three-finger W), the 23 mimed grasps of Set A (power and cylindrical "
             "wraps, hook, lateral, palmar, spherical, tip, tripod and the circular and prismatic "
             "precision family), and every step of the seven Set C sequences (one finger curled at a "
             "time, the adjacent pairs, the progressive fists, the thumb alone). The exclusion is by "
             "digit-state signature, so the ASL shapes that share one (S, U, H, W, F, O, M, N, 1, 2, 6, "
             "9) are also left out; `tests/test_pose_catalogue.py` checks it. The professor's 102 "
             "tracker frames carry no pose names and are not matched here.\n")
    L.append("## Recording setup\n")
    L.append(f"Every pose: {SETUP}. No gloves, no objects. One take is one held pose; record both "
             "hands if time allows, left first to match the pilot. The Leap column of the table says "
             "where the camera will struggle: a thumb hidden under the fingers is inferred, not seen, "
             "and curled fingertips can hide behind the proximal phalanx, so keep the palm square to "
             "the camera for those.\n")
    L.append("## The 100 poses\n")
    L.append("Each finger cell is `state: MCP, PIP, DIP, abd` in degrees; the thumb cell is MCP, IP and "
             "the TMC in words; the wrist cell is flexion and deviation.\n")
    L.append("| id | pose | index | middle | ring | little | thumb | wrist | Leap |\n|---|---|---|---|---|---|---|---|---|")
    for pose in plist:
        r = ranges(pose)
        thumb = (f"{pose.thumb}: MCP {_rng(r['T_MCP_fe'])}, IP {_rng(r['T_IP'])}; TMC "
                 f"{THUMB[pose.thumb][2]}" + (f" ({pose.thumb_target})" if pose.thumb_target else ""))
        wrist = f"{pose.wrist}: fe {_rng(r['WRIST_fe'])}, dev {_rng(r['WRIST_dev'])}"
        L.append(f"| {pose.id} | **{pose.name}**: {pose.description} | "
                 + " | ".join(_finger_cell(pose, f, r) for f in FINGERS)
                 + f" | {thumb} | {wrist} | {pose.leap_note()} |")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--doc", type=Path, default=DOC_PATH)
    args = ap.parse_args(argv)
    plist = poses()
    dup = [p.signature for p in plist if p.signature in DATABASE_SIGNATURES]
    ids = [p.id for p in plist]
    problems = []
    if len(plist) != 100:
        problems.append(f"{len(plist)} poses, not 100")
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    sigs = [p.signature for p in plist]
    if len(set(sigs)) != len(sigs):
        problems.append("duplicate signatures: " + ", ".join(s for s in sigs if sigs.count(s) > 1))
    if dup:
        problems.append("database poses: " + ", ".join(dup))
    for pose in plist:
        for col, (lo, hi) in ranges(pose).items():
            e = ENVELOPE[envelope_key(col)]
            if not (e[0] <= lo < hi <= e[1]):
                problems.append(f"{pose.id} {col} {lo}..{hi} outside {e}")
    if problems:
        print("POSE CATALOGUE FAIL: " + "; ".join(problems))
        return 1
    table = rows(plist)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(document(plist), encoding="utf-8")
    cats = {}
    for pose in plist:
        cats[pose.category] = cats.get(pose.category, 0) + 1
    print(f"wrote {args.csv}")
    print(f"wrote {args.doc}")
    print(f"POSE CATALOGUE OK: poses={len(plist)} categories="
          + ",".join(f"{k}={v}" for k, v in cats.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
