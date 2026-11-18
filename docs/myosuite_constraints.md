# MuJoCo and MyoSuite: what the two links a colleague sent can do for us

A colleague sent https://mujoco.org/ and https://github.com/MyoHub/myosuite on
6 Nov 2026, suggesting that they could complement the Leap Motion research:
captured hand movements could be compared with simulated biomechanical
models, and perhaps the constraints too. This note records what was tried on
this machine on 7 Nov 2026, what
came out, and what the two tools can and cannot do for the glove-and-camera
project. Every MyoSuite number below is read from the model by
`scripts/leap/myosuite_hand_limits.py` and is in
`results/myosuite_constraints/`; nothing is quoted from a web page.

## What the two tools are

**MuJoCo** is a physics engine: rigid bodies, joints with limits, contacts,
tendons and muscle actuators, with Python bindings (`pip install mujoco`,
version 4.0.0 here). It is the engine, not a hand.

**MyoSuite** is a library on top of MuJoCo. Its hand, **MyoHand**, is a
musculoskeletal right hand with 23 joints and 39 muscle actuators, converted
from two OpenSim models (the MoBL upper-extremity model and the 2nd-Hand
model, Holzbaur et al. 2005, as its hand README says). Since MyoSuite 3.0 the
model lives in a separate package, `myo_sim`, which composes the hand from its
arm model in memory; `myo_sim.load_model("myohand_r")` compiles it. MyoSuite
itself adds 24 `myoHand*` reinforcement-learning tasks (pose reaching, key
turn, pen twirl, baoding balls, die reorientation) and the Gymnasium wrapper
around them.

Both installed and ran on this Windows 11 machine with Python 3.14, in
`.venv-myo` in this repo: the hand compiles, 100 simulation steps leave every
joint finite, and `myoHandPoseFixed-v0` builds, resets and steps in 1.4 s
(observation 108 values, action 39 muscles). No Linux, no GPU and no PyTorch
were needed for this.

## What we can do with them

1. **A second limit table for an anatomical filter.** The second paper you
   sent the colleague clamps each tracked joint angle to its biomechanical limit.
   MyoHand carries one such table (its `jnt_range`), and this repo already
   carries another, Paper 3's Table 5 (`leap_hand.paper3.TABLE5`). Both are now
   side by side under our 24 angle names in
   `results/myosuite_constraints/comparison_24_angles.csv`, with the span the
   glove actually produced in the pilot takes as a third column. The filter
   can be written against either; the table below shows they do not agree,
   so choosing is a decision, not a lookup.

2. **Drive the simulated hand with our captured angles.** MyoHand's 23 joints
   are set directly (`data.qpos`), so a recorded take's angles can be written
   into the model frame by frame. That gives three things we do not have
   today: a 3D rendered hand for the replay viewer that obeys joint limits
   and self-collision, a fingertip-position comparison against the camera's
   fingertips of the kind Paper 3 reports in its Table 8, and a
   self-penetration check (MuJoCo contacts between finger geoms) as a
   plausibility test for a pose. The mapping from our 24 angles to MyoHand's
   signs and zero posture has to be built first; the sign conventions found so
   far are in the comparison section.

3. **Simulated grasps as a reference.** MyoSuite's tasks put the hand on a
   die, a pen, a key and two baoding balls. A trained policy's joint angles
   while holding those objects are a synthetic reference posture for the
   matching grasps in our protocol. This is the weakest of the three uses:
   those postures come from a policy's reward, not from a human, and the
   objects are not ours.

4. **Coupling relations.** MyoHand's muscles are tendon-driven (FDP, FDS, EDC
   and the intrinsics), so pulling one muscle moves PIP and DIP together.
   Paper 3's PIP ~ 3/2 DIP relation, which the replay layer already measures
   on our recordings, can be measured on the simulated hand the same way.

## What we cannot do with them

- **Treat MyoHand's limits as anatomy.** They are a model's `jnt_range`, and
  for the fingers they are round numbers: every MCP, PIP and DIP flexion runs
  0 to 90 degrees with no hyperextension, and every MCP abduction is plus or
  minus 15 degrees. Paper 3's literature ranges give PIP 110 to 135 degrees
  and 30 to 40 degrees of MCP hyperextension. A filter built from MyoHand
  would clip a real fist. Nothing here checks either table against anatomy
  sources.
- **Fill the gaps the camera leaves.** MyoHand has no finger CMC joints: its
  four finger metacarpals are fixed in the palm, exactly as LeapC's rigid
  palm is. The palm arc stays unmeasured and unsimulated.
- **Get human data.** MyoSuite ships models and tasks, not recordings of
  people. The comparison datasets in `comparison data sources 2026-10-02.md`
  remain the only human references.
- **Left hand by name.** `myo_sim.load_model` builds the right hand;
  a left-hand builder exists (`FRAGMENT_SPEC_BUILDERS["myohand_l"]`) but was
  not run here. Most pilot takes are left-handed.
- **Untested here:** MuJoCo's interactive viewer on this machine, and the
  tasks that want PyTorch (the "visual keys" and MyoMimic policies). Neither
  is needed for the uses above.

## What the joint comparison shows

The table is `comparison_24_angles.csv` rounded to whole degrees. MyoHand's
column is the joint's declared range in its own sign convention. Paper 3's
column is Table 5 as printed (the row's own range, then its hyperextension),
which gives no signed minimum or zero. The glove column is the span the
glove's exported angles covered across the left-hand open-palm, fist and
pinch takes of `recordings/sync_day2/` (the pilot audit's
`glove_angle_ranges.csv`); it is what three takes produced, not a limit.

Sign conventions, measured by setting each joint to its minimum and maximum
and reading the thumb-tip or fingertip geom's distance to the little-finger
MCP and the middle metacarpal: for the four thumb joints the **negative**
end brings the thumb toward the palm, so negative is flexion (and, for
`cmc_abduction_r`, adduction); for the finger joints the **positive** end is
flexion and abduction has no fixed sign. So MyoHand's thumb allows 45 degrees
of TMC flexion and 40 of extension, 29 of TMC adduction and 45 of abduction,
45 of MCP flexion and 40 of extension, 75 of IP flexion and 25 of extension.

| angle | MyoHand range (deg, model sign) | Paper 3 Table 5 (range; hyperextension) | glove span seen (deg) | note |
|---|---|---|---|---|
| T_TMC_fe | -45 to 40 (`cmc_flexion_r`) | 50–90; 15 | 2 to 39 | negative = flexion; other axes than the paper's, not convertible |
| T_TMC_aa | -29 to 45 (`cmc_abduction_r`) | 45–60; – | 28 to 35 | negative = adduction |
| T_MCP_fe | -45 to 40 (`mp_flexion_r`) | 75–80; 0 | 2 to 49 | negative = flexion; MyoHand allows 45 flexion against the paper's 75–80 |
| T_IP | -75 to 25 (`ip_flexion_r`) | 75–80; 5–10 | 2 to 71 | negative = flexion; 25 extension against the paper's 5–10 |
| I_CMC | no joint | 5; 0 | 0 to 0 | neither MyoHand nor the camera has a finger CMC |
| I_MCP_fe | 0 to 90 (`mcp2_flexion_r`) | 90; 30–40 | 0 to 90 | no hyperextension in MyoHand |
| I_MCP_aa | -15 to 15 (`mcp2_abduction_r`) | 60; – | 0 to 1 | the glove's index abduction hardly moves |
| I_PIP | 0 to 90 (`pm2_flexion_r`) | 110; 0 | 4 to 90 | MyoHand 20 short of the paper; glove tops out at exactly 90.00 |
| I_DIP | 0 to 90 (`md2_flexion_r`) | 80–90; 5 | 4 to 43 | |
| M_CMC | no joint | 5; 0 | 0 to 0 | |
| M_MCP_fe | 0 to 90 (`mcp3_flexion_r`) | 90; 30–40 | 0 to 81 | |
| M_MCP_aa | -15 to 15 (`mcp3_abduction_r`) | 45; – | 0 to 0 | glove shows no middle abduction |
| M_PIP | 0 to 90 (`pm3_flexion_r`) | 110; 0 | 4 to 100 | the glove exceeds MyoHand's limit; tops out at exactly 100.00 |
| M_DIP | 0 to 90 (`md3_flexion_r`) | 80–90; 5 | 5 to 40 | |
| R_CMC | no joint | 10; 0 | 0 to 0 | |
| R_MCP_fe | 0 to 90 (`mcp4_flexion_r`) | 90; 30–40 | 0 to 88 | |
| R_MCP_aa | -15 to 15 (`mcp4_abduction_r`) | 45; – | -3 to 1 | |
| R_PIP | 0 to 90 (`pm4_flexion_r`) | 120; 0 | 4 to 100 | the glove exceeds MyoHand's limit |
| R_DIP | 0 to 90 (`md4_flexion_r`) | 80–90; 5 | 6 to 50 | |
| L_CMC | no joint | 15; 0 | 0 to 0 | |
| L_MCP_fe | 0 to 90 (`mcp5_flexion_r`) | 90; 30–40 | 0 to 93 | the glove exceeds MyoHand's limit by 3 |
| L_MCP_aa | -15 to 15 (`mcp5_abduction_r`) | 50; – | -6 to 0 | |
| L_PIP | 0 to 90 (`pm5_flexion_r`) | 135; 0 | 5 to 90 | MyoHand 45 short of the paper |
| L_DIP | 0 to 90 (`md5_flexion_r`) | 90; 5 | 8 to 65 | |

What this says:

- **The two limit tables disagree where it matters for a fist.** MyoHand
  caps PIP at 90 degrees for every finger; the literature ranges in Paper 3
  run 110 to 135, and the glove itself reported 100 on the middle and ring
  PIP during the fist take. Clamping to MyoHand would cut real flexion.
- **MyoHand allows no MCP hyperextension**; Paper 3 allows 30 to 40 degrees,
  and the open-palm take needs some of it.
- **MyoHand's abduction range (30 degrees total) is far below the paper's**
  45 to 60; the glove showed almost none, which is the glove's known spread
  limitation, not evidence either way.
- **The glove's own exports look clamped**: index PIP stops at exactly 90.00,
  middle and ring PIP at exactly 100.00, index MCP at 89.76 just under 90.
  Round-number ceilings like these are software limits in the glove's export,
  not joints stopping. That is worth confirming before any limit comparison
  uses glove maxima.
- **The thumb cannot be compared cell by cell.** MyoHand's thumb axes, like
  the camera's, are not the paper's TMC axes (`docs/tmc_estimate.md`), and
  its ranges are two-sided around a zero posture the paper does not define.

## How to reproduce

One command, in the MyoSuite venv (made once with `py -m venv .venv-myo` and
`.venv-myo\Scripts\python.exe -m pip install mujoco myosuite myo_sim`; this
machine has mujoco 4.0.0, myosuite 3.0.0, myo_sim 0.2.3):

    .venv-myo\Scripts\python.exe scripts\leap\myosuite_hand_limits.py

It prints `MYOHAND OK: ... joints=23 steps=100 finite=yes` and writes
`results/myosuite_constraints/myohand_joint_limits.csv` (all 23 joints) and
`results/myosuite_constraints/comparison_24_angles.csv` (the table above,
unrounded). `--check` runs only the load-and-step test; `--list-joints`
prints every joint. `tests/test_myosuite_constraints.py` checks the CSVs and
this note in the repo venv. The gates are in `GATES_myosuite.md`. Composing
the hand prints one MuJoCo warning ("Attaching a spec with an empty prefix")
and MuJoCo writes it to `MUJOCO_LOG.TXT` in the working directory; the file
is harmless and ignored by git.

## Next step

Decide with the colleague which table the anatomical filter clamps to. The evidence
here points to Paper 3's Table 5 for the limits (it is the literature table,
it is already in the repo, and MyoHand's finger limits are tighter than real
fists) and to MyoHand for what a table cannot do: driving a simulated hand
with our angles to render the replay, check self-penetration and measure
fingertip error. The first concrete task on that path is the mapping from our
24 angles to MyoHand's `qpos` (zero posture and the thumb's sign flips, from
the convention measured above), then one recorded fist replayed through it.
Before any glove maximum is used as a limit, confirm with StretchSense whether
the 90.00 and 100.00 ceilings are export clamps.
