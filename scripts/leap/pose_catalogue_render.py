r"""Render every catalogue pose on MyoSuite's MyoHand, two views per pose.

Not photographs: none of the 100 poses has been recorded yet. Each image is
the MyoHand model (myo_sim, right hand) posed at the midpoint of the
catalogue's template range for every camera-given angle, with the thumb's
TMC set from the pose's thumb state in words (and from the finger it is
aimed at, when the pose names one). Joint values are clamped to MyoHand's
own ranges, so a full curl shows PIP 90, not the catalogue's 85 to 110,
with one exception: the finger MCP is allowed back to -40 for the picture,
so a finger pressed back past straight is drawn that way (MyoHand itself
stops at straight).

Runs in the MyoSuite venv:

    .venv-myo\Scripts\python.exe scripts\leap\pose_catalogue_render.py [--only A01,B03] [--size 420]

writes results/pose_catalogue/renders/<id>_palm.png (palm toward the viewer,
as the Leap sees it) and <id>_side.png (from the thumb side), and
renders_index.csv listing every file with the clamped joint values.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "pose_catalogue" / "renders"


def load_catalogue():
    spec = importlib.util.spec_from_file_location("pose_catalogue", REPO / "scripts" / "leap" / "pose_catalogue.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pose_catalogue"] = mod
    spec.loader.exec_module(mod)
    return mod


# Thumb TMC in MyoHand coordinates for each thumb state: (cmc_flexion_r, cmc_abduction_r) radians.
# MyoHand: negative cmc_flexion = flexion across the palm, negative cmc_abduction = adduction.
TMC = {
    "out": (0.30, 0.60), "up": (0.45, 0.10), "hyper_up": (0.65, 0.10),
    "along": (0.20, -0.35), "across": (-0.60, -0.20), "tucked": (-0.70, -0.40),
    "between": (-0.45, -0.25), "oppose": (-0.50, 0.25), "side": (-0.40, -0.10),
    "c_curve": (-0.10, 0.55), "ip_bent": (0.30, 0.60), "mcp_bent": (0.30, 0.60),
}
# Where the thumb points when the pose names a finger for it: (thumb state, target) -> TMC.
# Chosen by eye like the table above, so that "thumb to index base" and "thumb to little
# base" are not the same picture; the catalogue has no TMC range to take them from.
TMC_TARGET = {
    ("side", "index base"): (-0.10, -0.35), ("side", "middle base"): (-0.33, -0.18),
    ("side", "ring base"): (-0.56, 0.00), ("side", "little base"): (-0.78, 0.15),
    ("oppose", "index"): (-0.35, 0.30), ("oppose", "middle"): (-0.50, 0.25),
    ("oppose", "ring"): (-0.65, 0.20), ("oppose", "little"): (-0.78, 0.15),
}
FINGER_NUM = {"I": 2, "M": 3, "R": 4, "L": 5}
# MyoHand stops the finger MCP at 0 (straight). For the picture only, the lower limit is
# opened to the catalogue's envelope (-40), so a finger pressed back is drawn pressed back.
RENDER_LOWER = {f"mcp{n}_flexion_r": math.radians(-40) for n in FINGER_NUM.values()}


def mid(r):
    return (r[0] + r[1]) / 2.0


def joint_targets(mod, pose) -> dict:
    """MyoHand joint name -> target radians for one pose (before clamping)."""
    r = mod.ranges(pose)
    q = {}
    for f, n in FINGER_NUM.items():
        q[f"mcp{n}_flexion_r"] = math.radians(mid(r[f"{f}_MCP_fe"]))
        q[f"mcp{n}_abduction_r"] = math.radians(mid(r[f"{f}_MCP_aa"]))
        q[f"pm{n}_flexion_r"] = math.radians(mid(r[f"{f}_PIP"]))
        q[f"md{n}_flexion_r"] = math.radians(mid(r[f"{f}_DIP"]))
    # thumb: MyoHand flexes toward negative values
    q["mp_flexion_r"] = -math.radians(mid(r["T_MCP_fe"]))
    q["ip_flexion_r"] = -math.radians(mid(r["T_IP"]))
    q["cmc_flexion_r"], q["cmc_abduction_r"] = TMC_TARGET.get((pose.thumb, pose.thumb_target), TMC[pose.thumb])
    # wrist: flexion positive toward the palm in both; deviation positive toward the thumb
    # in the catalogue, positive toward the little finger in MyoHand
    q["flexion_r"] = math.radians(mid(r["WRIST_fe"]))
    q["deviation_r"] = -math.radians(mid(r["WRIST_dev"]))
    q["pro_sup_r"] = 0.0
    return q


def apply(model, data, targets: dict) -> dict:
    import mujoco
    clamped = {}
    mujoco.mj_resetData(model, data)
    for name, val in targets.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            continue
        lo, hi = model.jnt_range[j]
        v = min(max(val, min(lo, RENDER_LOWER.get(name, lo))), hi)
        data.qpos[model.jnt_qposadr[j]] = v
        clamped[name] = round(math.degrees(v), 1)
    mujoco.mj_forward(model, data)
    return clamped


def cameras(model, data):
    """Palm view (camera on the palm normal, as the Leap sees a hand held over it) and side view."""
    import mujoco

    def pos(name):
        return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
    wrist, idx, lit, mid3 = pos("lunate_r"), pos("2proxph_r"), pos("5proxph_r"), pos("3proxph_r")
    n = np.cross(idx - wrist, lit - wrist)
    n /= np.linalg.norm(n)
    lookat = (wrist + mid3) / 2 + 0.02 * (mid3 - wrist) / np.linalg.norm(mid3 - wrist)
    # palmar side: curl every finger and keep the camera position nearer the fingertips
    saved = data.qpos.copy()
    for jn in ("mcp3_flexion_r", "pm3_flexion_r", "md3_flexion_r"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[j]] = 1.3
    mujoco.mj_forward(model, data)
    tip = pos("distph3_r")
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    if np.linalg.norm(lookat - n * 0.3 - tip) < np.linalg.norm(lookat + n * 0.3 - tip):
        n = -n
    fwd = -n                                      # camera looks against the palm normal
    az = math.degrees(math.atan2(fwd[1], fwd[0]))
    el = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
    palm = mujoco.MjvCamera()
    palm.type = mujoco.mjtCamera.mjCAMERA_FREE
    palm.lookat[:] = lookat
    palm.distance, palm.azimuth, palm.elevation = 0.30, az, el
    # Verify with the real camera position: it must be nearer the curled fingertip than the
    # metacarpal head; otherwise the camera is on the back of the hand, so turn it around.
    scn = mujoco.MjvScene(model, maxgeom=2000)
    for jn in ("mcp3_flexion_r", "pm3_flexion_r", "md3_flexion_r"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[j]] = 1.3
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, palm, mujoco.mjtCatBit.mjCAT_ALL, scn)
    cam_pos = np.array(scn.camera[0].pos)
    tip, head = pos("distph3_r"), pos("3proxph_r")
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    if np.linalg.norm(cam_pos - tip) > np.linalg.norm(cam_pos - head):
        palm.azimuth, palm.elevation = az + 180.0, -el
        n = -n
    # side view: from the thumb side, in the palm plane, looking across the hand
    side_dir = np.cross(n, (mid3 - wrist) / np.linalg.norm(mid3 - wrist))   # +/- thumb side
    thumb = pos("firstmc_r")
    if np.dot(thumb - wrist, side_dir) < 0:
        side_dir = -side_dir
    fwd2 = -side_dir
    side = mujoco.MjvCamera()
    side.type = mujoco.mjtCamera.mjCAMERA_FREE
    side.lookat[:] = lookat
    side.distance = 0.30
    side.azimuth = math.degrees(math.atan2(fwd2[1], fwd2[0]))
    side.elevation = math.degrees(math.asin(max(-1.0, min(1.0, fwd2[2]))))
    return palm, side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated pose ids")
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    import mujoco
    import myo_sim
    from PIL import Image

    mod = load_catalogue()
    plist = mod.poses()
    if args.only:
        want = set(args.only.split(","))
        plist = [p for p in plist if p.id in want]
    model = myo_sim.load_model("myohand_r")
    data = mujoco.MjData(model)
    # cameras from the neutral hand, so every pose is framed the same way
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cam_palm, cam_side = cameras(model, data)
    # show only the hand and the end of the forearm; everything else transparent
    hand_bodies = {"radius_r", "ulna_r"}
    chain_start = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lunate_r")
    for b in range(model.nbody):
        anc = b
        while anc > 0 and anc != chain_start:
            anc = int(model.body_parentid[anc])
        if anc == chain_start:
            hand_bodies.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
    for g in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) not in hand_bodies:
            model.geom_rgba[g, 3] = 0.0
    renderer = mujoco.Renderer(model, args.size, args.size)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    depth = mujoco.Renderer(model, args.size, args.size)
    depth.enable_depth_rendering()
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    args.out.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for pose in plist:
        clamped = apply(model, data, joint_targets(mod, pose))
        for view, cam in (("palm", cam_palm), ("side", cam_side)):
            renderer.update_scene(data, cam, opt)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            img = renderer.render().copy()
            depth.update_scene(data, cam, opt)
            far = depth.render() > 2.0            # metres; nothing of the hand is that far
            img[far] = 255
            path = args.out / f"{pose.id}_{view}.png"
            Image.fromarray(img).save(path)
        index_rows.append({"id": pose.id, "name": pose.name, "palm_png": f"{pose.id}_palm.png",
                           "side_png": f"{pose.id}_side.png",
                           **{f"q_{k}": v for k, v in clamped.items()}})
    with (args.out / "renders_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        w.writeheader()
        w.writerows(index_rows)
    print(f"RENDERS OK: poses={len(plist)} files={2 * len(plist)} dir={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
