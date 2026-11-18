r"""MyoSuite's hand joint ranges, in degrees, beside Paper 3 and the glove.

Loads MyoSuite's musculoskeletal hand, MyoHand (23 joints, 39 muscles; the
`myo_sim` package composes the right hand from its arm model and MuJoCo
compiles it), reads every joint's range as the model declares it, and writes
two CSVs:

  results/myosuite_constraints/myohand_joint_limits.csv
      one row per joint of the MyoSuite hand model: name, type, limited,
      min_deg, max_deg, span_deg, axis, parent body.
  results/myosuite_constraints/comparison_24_angles.csv
      the repo's 24 paper angles (xr_hand.joint_frames.DOF24_NAMES, in order),
      each with the MyoSuite joint it corresponds to and that joint's range,
      Paper 3's Table 5 range and hyperextension as printed
      (leap_hand.paper3.TABLE5), and the glove's measured span from the pilot
      audit's glove_angle_ranges.csv. A cell is empty where a source has
      nothing for that angle.

Runs in the MyoSuite venv (`.venv-myo`: `py -m venv .venv-myo` then
`.venv-myo\Scripts\python.exe -m pip install mujoco myosuite myo_sim`), which
has mujoco, myosuite, myo_sim and numpy but not the rest of this repo's
dependencies; the repo's `src` is put on the path for the two pure-python
tables it needs.

    .venv-myo\\Scripts\\python.exe scripts\\leap\\myosuite_hand_limits.py --check
    .venv-myo\\Scripts\\python.exe scripts\\leap\\myosuite_hand_limits.py

`--check` only loads the model, steps it 100 times and prints the OK line.
The ranges printed are what the model declares (`jnt_range`, the arm chain's
`range` attributes in radians, converted); myo_sim's hand README derives the
model from the MoBL upper-extremity and 2nd-Hand OpenSim models (Holzbaur et
al. 2005), and nothing here verifies the ranges against anatomy sources.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from xr_hand.joint_frames import DOF24_NAMES, DOF24_SOURCE  # noqa: E402
from leap_hand.paper3 import TABLE5  # noqa: E402

DEFAULT_OUT = REPO / "results" / "myosuite_constraints"
DEFAULT_GLOVE = (REPO / "results" / "glove_camera_pilot_20261025" / "historical"
                 / "glove_angle_ranges.csv")

# The MyoSuite hand's joint for each of the paper's 24 angles. MyoSuite names
# the digits 1 (thumb) to 5 (little); its finger joints are mcp (flexion and
# abduction), pm (proximal phalanx, our PIP) and md (middle-distal, our DIP);
# its thumb has cmc (our TMC), mp (our MCP) and ip. The composed right hand
# suffixes every joint with `_r`; the match below ignores that suffix. It has
# no finger CMC joints: its metacarpals are fixed in the palm, as LeapC's are.
MYO_JOINT = {
    "T_TMC_fe": "cmc_flexion", "T_TMC_aa": "cmc_abduction",
    "T_MCP_fe": "mp_flexion", "T_IP": "ip_flexion",
}
for _letter, _n in (("I", 2), ("M", 3), ("R", 4), ("L", 5)):
    MYO_JOINT.update({
        f"{_letter}_CMC": "",
        f"{_letter}_MCP_fe": f"mcp{_n}_flexion",
        f"{_letter}_MCP_aa": f"mcp{_n}_abduction",
        f"{_letter}_PIP": f"pm{_n}_flexion",
        f"{_letter}_DIP": f"md{_n}_flexion",
    })
assert set(MYO_JOINT) == set(DOF24_NAMES)


def load_hand():
    """MyoHand (right) compiled by MuJoCo, and the name myo_sim gives it."""
    import myo_sim
    return myo_sim.load_model("myohand_r"), f"myo_sim {myo_sim.__version__} myohand_r"


def unsuffixed(name: str) -> str:
    return name[:-2] if name.endswith(("_r", "_l")) else name


def joint_rows(model) -> list[dict]:
    import mujoco
    types = {mujoco.mjtJoint.mjJNT_HINGE: "hinge", mujoco.mjtJoint.mjJNT_SLIDE: "slide",
             mujoco.mjtJoint.mjJNT_BALL: "ball", mujoco.mjtJoint.mjJNT_FREE: "free"}
    rows = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        jtype = types.get(model.jnt_type[j], str(model.jnt_type[j]))
        limited = bool(model.jnt_limited[j])
        lo, hi = (float(v) for v in model.jnt_range[j])
        angular = jtype in ("hinge", "ball")
        if limited and angular:
            lo_deg, hi_deg = math.degrees(lo), math.degrees(hi)
            span = hi_deg - lo_deg
        elif limited:
            lo_deg, hi_deg, span = lo, hi, hi - lo  # metres for a slide
        else:
            lo_deg = hi_deg = span = float("nan")
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.jnt_bodyid[j])) or ""
        rows.append({
            "joint": name, "type": jtype, "limited": "yes" if limited else "no",
            "min_deg": "" if math.isnan(lo_deg) else f"{lo_deg:.2f}",
            "max_deg": "" if math.isnan(hi_deg) else f"{hi_deg:.2f}",
            "span_deg": "" if math.isnan(span) else f"{span:.2f}",
            "axis": " ".join(f"{float(a):.3f}" for a in model.jnt_axis[j]),
            "body": body,
        })
    return rows


def load_glove_spans(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(joint, component) -> (min_deg, max_deg) from the pilot audit's CSV."""
    if not path.is_file():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["joint"], row["component"])] = (row["min_deg"], row["max_deg"])
    return out


def comparison_rows(joints: list[dict], glove: dict) -> list[dict]:
    by_name = {unsuffixed(r["joint"]): r for r in joints}
    rows = []
    for name in DOF24_NAMES:
        myo = MYO_JOINT[name]
        jr = by_name.get(myo) if myo else None
        t5 = TABLE5[name]
        gkey = DOF24_SOURCE[name]
        g = glove.get(gkey, ("", ""))

        def fmt(v: str) -> str:
            return f"{float(v):.2f}" if v not in ("", None) else ""

        rows.append({
            "angle": name,
            "myosuite_joint": jr["joint"] if jr else "",
            "myosuite_min_deg": jr["min_deg"] if jr else "",
            "myosuite_max_deg": jr["max_deg"] if jr else "",
            "myosuite_span_deg": jr["span_deg"] if jr else "",
            "paper3_table5_range": (t5["abduction_range"] if name.endswith("_aa")
                                    else t5["flexion_range"]) or "",
            "paper3_table5_hyperextension": t5["hyperextension"] or "",
            "glove_row": f"{gkey[0]}.{gkey[1]}",
            "glove_min_deg": fmt(g[0]),
            "glove_max_deg": fmt(g[1]),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="load, step 100 times, print the OK line; write nothing")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--glove-ranges", type=Path, default=DEFAULT_GLOVE)
    ap.add_argument("--list-joints", action="store_true",
                    help="print every joint name after the OK line")
    args = ap.parse_args(argv)

    import mujoco
    import myosuite
    model, source = load_hand()
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    finite = bool(np.all(np.isfinite(data.qpos)))
    rows = joint_rows(model)
    print(f"MYOHAND OK: mujoco={mujoco.__version__} myosuite={myosuite.__version__} "
          f"joints={len(rows)} steps=100 finite={'yes' if finite else 'no'}")
    print(f"model: {source}")
    if args.list_joints:
        for r in rows:
            print(f"  {r['joint']:20s} {r['type']:5s} limited={r['limited']:3s} "
                  f"{r['min_deg']:>8s} {r['max_deg']:>8s}  body={r['body']}")
    if args.check:
        return 0 if finite else 1

    write_csv(args.out / "myohand_joint_limits.csv", rows)
    comp = comparison_rows(rows, load_glove_spans(args.glove_ranges))
    write_csv(args.out / "comparison_24_angles.csv", comp)
    unmatched = [c["angle"] for c in comp if MYO_JOINT[c["angle"]] and not c["myosuite_joint"]]
    print(f"wrote {args.out / 'myohand_joint_limits.csv'}")
    print(f"wrote {args.out / 'comparison_24_angles.csv'} rows={len(comp)} "
          f"glove_source={'present' if args.glove_ranges.is_file() else 'missing'}")
    if unmatched:
        print(f"WARNING: no MyoSuite joint named for {unmatched}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
