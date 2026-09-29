# Joint frames: every joint's x y z axes and the paper's 24 angles

Cobos et al. 2009 ("Human Hand Descriptions and Gesture Recognition for
Object Manipulation", Figure 1, Figure 3 and Table 1) draw a local x y z frame
at every joint of the hand and name the angles between them. This page is how
to see the same thing on our data: the 26 OpenXR joints of the glove and of
the Ultraleap camera, each with its own axes drawn on the hand, and the
numbers behind them.

The numbers come from `src/xr_hand/joint_frames.py`; the viewer and the
exports are `scripts/joint_frames_view.py`.

## What the numbers are

For one frame of a take, 26 rows, one per joint, in the order of
`xr_hand.joints.JOINT_NAMES` (PALM, WRIST, then thumb and each finger from
the metacarpal to the tip):

| Column | Meaning |
|---|---|
| `joint`, `parent` | the joint and the joint it hangs from (empty for WRIST) |
| `x_mm y_mm z_mm` | the joint's position in the **wrist frame**, millimetres |
| `qx qy qz qw` | the joint's orientation in the wrist frame (quaternion, w >= 0) |
| `axis_x_*`, `axis_y_*`, `axis_z_*` | the joint's own three unit axes, as vectors in the wrist frame (`axis_x_y` = the y component of the joint's x axis) |
| `bone_len_mm` | distance to the parent joint |
| `flex_deg abd_deg twist_deg` | the rotation from the parent joint's frame to this joint's frame |

The **wrist frame** has its origin at the wrist joint and the wrist joint's
own axes. It is the one frame in which the glove (which has no position in
the room) and the camera can be put side by side.

## Axes and signs

Every joint's frame follows one rule on both sensors: z runs along the bone,
pointing back toward the wrist; y points out of the back of the hand; x goes
across (x = y cross z).

- Glove: each bone lies along -z of its parent, +y is the back of the hand,
  and every finger joint turns about its own x axis only (a day-2 fist
  reads -90 degrees about x at the index PIP).
- Camera: LeapC's bone frames, whose -z runs from a joint to the next one,
  +y is the back of the hand; the WRIST joint's orientation is LeapC's
  forearm bone.

The rotation from the parent's frame to the child's is split as x first,
then the new y, then the new z:

| Angle | About | Positive means |
|---|---|---|
| `flex_deg` | x | toward the palm (a curling finger) |
| `abd_deg` | y | toward the thumb side of the hand |
| `twist_deg` | z, the bone | the back of the bone turns toward the little finger |

The signs are the same for the left and the right hand and for both sensors.
x first means abduction is the angle that could lock, at 90 degrees, which no
finger reaches; a fist's 90 degrees of flexion stays clean. The reasons are
written out in the module's docstring.

One thing to keep in mind with the camera: because its wrist joint turns with
the forearm, a bent wrist turns the whole camera hand in the wrist frame, and
the four finger CMC angles include that bend. The PDF and the live window
print the forearm-to-palm angle on every camera hand (on day 2 it was 26 to
39 degrees on the left-hand takes). `--frame-ref palm` uses the palm's axes
instead; the glove's numbers do not change with it.

## How to read the 24 names

The paper's 24 angles are cells of the table:

| Name | Row | Column |
|---|---|---|
| `T_TMC_fe` | THUMB_METACARPAL | flex |
| `T_TMC_aa` | THUMB_METACARPAL | abd |
| `T_MCP_fe` | THUMB_PROXIMAL | flex |
| `T_IP` | THUMB_DISTAL | flex |
| `I_CMC`, `M_CMC`, `R_CMC`, `L_CMC` | the finger's METACARPAL | flex |
| `I_MCP_fe` ... `L_MCP_fe` | the finger's PROXIMAL | flex |
| `I_MCP_aa` ... `L_MCP_aa` | the finger's PROXIMAL | abd |
| `I_PIP` ... `L_PIP` | the finger's INTERMEDIATE | flex |
| `I_DIP` ... `L_DIP` | the finger's DISTAL | flex |

T thumb, I index, M middle, R ring, L little. A row names the joint at the
start of its bone, so the PROXIMAL row holds the MCP joint's angles.

- The thumb's TMC angles are measured from the wrist frame, so they include
  the thumb's resting position: they are not zero with the hand open. Its
  twist (in the CSV) is the thumb metacarpal's roll, which is where
  opposition shows.
- The glove senses flexion only. Its CMC, MCP_aa and thumb TMC values come
  from its own hand model (the metacarpal splay is a fixed 5, 10, 15
  degrees); the camera measures them.

Measured on day 2 (`recordings\sync_day2`, the medoid frame of each take),
index PIP flexion: glove fist 89.8 to 90.0, glove open palm 4.1, camera fist
77.3 to 80.3, camera open palm 3.6 to 5.3.

## The three commands

Every command runs from the repo folder:

```
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
```

### Camera only, bare hand: see the joint frames live

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --live
```

The window shows the IR picture with a 1 cm triad at every joint (x red,
y green, z blue) and, on the right, the 26 rows (x y z in mm, flexion,
abduction) and the 24 angles. Keys:

| Key | What it does |
|---|---|
| n | highlights the next finger (rows and axes) |
| s | saves a PNG of the window into `recordings\joint_frames\` |
| q or Esc | closes the window |

It is a read-only client of the tracking service, like the camera window of
the recorders, so it can stay open while a recorder runs.

With the gloves on and XR Trainer streaming (gloves on + camera), add
`--glove`: the glove hand is drawn at the camera hand's wrist and `g` switches
the table between the camera hand and the glove hand. Do not add `--glove`
while a glove recorder is running: both need the glove's network port.

### No hardware: one take

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --take recordings\sync_day2\glove\fist_left_take1_20260920_195141.jsonl
```

It picks the take's medoid frame (the real frame closest to the take's
average, the rule of `scripts\leap\record_frame.py`), writes
`<take>.csv` and `<take>.png` beside the take, and prints the 24 angles. The
PNG is a drawing on a white background (no camera pixels): the hand seen
from the palm side and from the side, the axes at every joint, the paper's
joint names, and the 24 angles. `--frame 40` uses line 40 of the file
(counting from 0) instead, `--out <folder>` writes somewhere else.

### No hardware: a whole session

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --session recordings\protocol\grasps\20260928_140501_left
```

Use your own session folder. For every accepted take it writes the summary
frame into `<session>\joint_frames\`: for Set A the medoid of the static
interval that the recorder chose, for Sets B and C the medoid of the whole
take, from the glove file (`<take>.csv`) and from the camera file
(`camera_<take>.csv`). It also writes one `joint_frames.pdf` with a page per
summary frame: the drawing and the 26-row table.

The PDF is what goes to the professor: `scripts\package_professor_set.py`
copies the CSVs and the PDF into the hand-in folder and never the PNGs,
because the hand-in takes no image files.
