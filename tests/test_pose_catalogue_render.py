"""The catalogue's pose pictures: which way the render script tilts the wrist.

scripts/leap/pose_catalogue_render.py needs mujoco and myo_sim, which are only
in the MyoSuite venv (.venv-myo), and that venv has no pytest. So the test runs
a probe in that interpreter and checks what the probe measured on the posed model.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "leap" / "pose_catalogue_render.py"
MYO_PYTHON = REPO / ".venv-myo" / "Scripts" / "python.exe"

# Poses every catalogue pose whose wrist deviation range excludes zero, with the
# script's own joint_targets and apply, and measures the tilt on the model's bodies.
# The thumb side is the direction from the little-finger knuckle to the index
# knuckle of the neutral hand, not anything the script says about signs.
PROBE = r"""
import importlib.util, json, math, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import mujoco, myo_sim

spec = importlib.util.spec_from_file_location("pose_catalogue_render", sys.argv[1])
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)
cat = render.load_catalogue()
model = myo_sim.load_model("myohand_r")
data = mujoco.MjData(model)

BODIES = {"wrist": "capitate_r", "index": "2proxph_r", "middle": "3proxph_r",
          "little": "5proxph_r", "thumb_tip": "distal_thumb_r"}

def hand():
    return {k: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)].copy()
            for k, b in BODIES.items()}

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
n = hand()
along = n["middle"] - n["wrist"]                 # the hand's long axis, wrist to middle knuckle
along /= np.linalg.norm(along)
thumb_side = n["index"] - n["little"]
thumb_side -= np.dot(thumb_side, along) * along
thumb_side /= np.linalg.norm(thumb_side)

poses = []
for pose in cat.poses():
    lo, hi = cat.ranges(pose)["WRIST_dev"]
    if lo <= 0 <= hi:
        continue
    render.apply(model, data, render.joint_targets(cat, pose))
    p = hand()
    axis = p["middle"] - p["wrist"]
    poses.append({
        "id": pose.id, "dev_min": lo, "dev_max": hi,
        "tilt_deg": math.degrees(math.atan2(np.dot(axis, thumb_side), np.dot(axis, along))),
        "knuckle_shift_mm": 1000 * float(np.dot(p["middle"] - n["middle"], thumb_side))})
print(json.dumps({"thumb_tip_mm": 1000 * float(np.dot(n["thumb_tip"] - n["middle"], thumb_side)),
                  "poses": poses}))
"""


def _measured() -> dict:
    if not MYO_PYTHON.is_file():
        pytest.skip(".venv-myo is not here: the render script needs its mujoco and myo_sim")
    done = subprocess.run([str(MYO_PYTHON), "-c", PROBE, str(SCRIPT)],
                          capture_output=True, text=True, cwd=REPO, timeout=300)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_wrist_deviation_is_drawn_toward_the_named_side():
    got = _measured()
    assert got["thumb_tip_mm"] > 0, "the model's thumb is not on the index side of its knuckles"
    # the catalogue's WRIST_dev is positive toward the thumb (radial)
    radial = [p for p in got["poses"] if p["dev_min"] > 0]
    ulnar = [p for p in got["poses"] if p["dev_max"] < 0]
    assert radial and ulnar, "the catalogue has no radial or no ulnar pose to draw"
    for p in radial:
        assert p["knuckle_shift_mm"] > 0, f"{p['id']} (radial) is drawn toward the little finger: {p}"
    for p in ulnar:
        assert p["knuckle_shift_mm"] < 0, f"{p['id']} (ulnar) is drawn toward the thumb: {p}"
    # and tilted as far as the pose asks, not stopped short by a limit of the model
    for p in radial + ulnar:
        assert p["dev_min"] <= p["tilt_deg"] <= p["dev_max"], \
            f"{p['id']} is drawn at {p['tilt_deg']:.1f} deg, the catalogue asks {p['dev_min']} to {p['dev_max']}"
