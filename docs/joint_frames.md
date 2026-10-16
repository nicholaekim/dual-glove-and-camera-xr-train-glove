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

One exception, measured on the camera: the thumb's bones are turned about
their length, so on its MCP and IP rows (THUMB_PROXIMAL, THUMB_DISTAL) a
positive `abd_deg` leans the bone toward the palm's face, not toward the
thumb side (on the 69 tables of the 2026-10-01 grasp session,
`xr_hand.joint_key`; section 12 of the hand-in README measures it again on
the folder's own tables at every build).

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

## The joint key

Which joint each of the 26 rows is, and which of the paper's angles it
carries, is `src/xr_hand/joint_key.py`. Its naming rule (`NAMING_RULE`): a
row is named after the bone that starts at its joint, and its angles are
the turn from the bone before it (the parent row's) to that bone. So on the
index, middle, ring and little finger `*_METACARPAL` = CMC, `*_PROXIMAL` =
MCP, `*_INTERMEDIATE` = PIP, `*_DISTAL` = DIP; on the thumb
`THUMB_METACARPAL` = TMC, `THUMB_PROXIMAL` = MCP, `THUMB_DISTAL` = IP;
PALM = the wrist (the palm against the forearm); WRIST = the root (the
forearm's frame, no angles); `*_TIP` = the fingertip, not a joint (angles
0).

For the grasp set, when the session has joint-frame tables, the packager
writes the key beside them in the hand-in folder:

| File | What it is |
|---|---|
| `grasps\joint_frames\joint_key.csv` | 26 rows in table order, columns `number` (1 to 26), `row`, `anatomical_joint`, `bone_before`, `bone_after`, `flex_deg_means`, `abd_deg_means`, `paper_angle_flex`, `paper_angle_abd`, `given` (`yes`, `no: not measured by this tracker` for the six of `CAMERA_NOT_GIVEN` below, or `not one of the 24`) and `landmark_in_text_files` (the joint's label in the 21-landmark keypoint files; the four finger CMC are not in them) |
| `grasps\joint_frames\joint_key.pdf` | one A4 page: a flat left hand seen from the palm, drawn by hand (not measured), its 26 joints numbered as the rows, and the table beside it |

README section 11 prints the same key (`joint_key.readme_lines`), and
`scripts\validate_grasp_set.py` checks the CSV's 26 rows and order, that
each given angle of the paper sits in the row and column the angle table
takes it from, the PDF's one page, and the README's copy.

## Where the 24 are not the paper's numbers

The names are the paper's. The numbers are read off a tracker's bone
orientations, and differ from the paper's joint variables in five ways
(checked against the paper on 2026-10-05; `xr_hand.joint_frames` has the
detail, and the professor's README states all of it). Of the 24, 18 can be
read as named; the four finger CMC are not measured on the camera at all;
the thumb's two TMC angles are about other axes than the paper's. Those
six (`xr_hand.joint_frames.CAMERA_NOT_GIVEN`) are therefore not handed in
under the paper's names: `joint_angles_24.csv` leaves them empty and every
camera page of the PDF prints "n/a" for them, with what the tracker does
measure at those joints under names that say what it is
(`*_metacarpal_flex_wrist_ref`, `*_metacarpal_flex_palm_ref`,
`T_metacarpal_abd_*`). The joint-frame tables are unchanged: a row is a
joint against its parent, whatever a paper calls it.

- Order at the two-axis joints. The paper's chain (its Tables 2 and 3) turns
  the MCP about the abduction axis first and the flexion axis second. The
  table here takes flexion first, because the paper's order is undefined at
  90 degrees of flexion, which every fist reaches. The bone points the same
  way either way; the two numbers differ once both are non-zero:
  sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe). 60
  degrees of flexion with 10 of abduction here is 58.5 with 19.4 in the
  paper's order. `joint_angles_24.csv` in the hand-in folder gives the four
  MCP pairs both ways (`*_paper_order`).
- The four finger CMC are not measured on the camera. The paper's CMC is
  the arching of the palm. LeapC's palm is rigid: measured from the PALM
  joint the four are the same numbers in every one of the 69 summary frames
  of the 2026-10-01 grasp session (8.65, 8.53, 8.59 and 6.29 degrees, index
  to little finger; the `*_metacarpal_flex_palm_ref` columns of
  `joint_angles_24.csv`). The METACARPAL rows' own flexion is that rigid
  palm seen from the WRIST joint, which on the camera is the forearm: it
  changes with the bend of the wrist and with nothing else (-60 to 79
  degrees over that session, against the paper's own range of 5 to 15
  degrees for a CMC). On the glove they are constants of the glove's own
  model as well.
- The thumb's TMC pair is about the hand's axes, not the thumb's. In the
  paper's chain (its Table 3) the TMC flexion axis is parallel to the
  thumb's MCP and IP flexion axes. Here TMC_fe and TMC_aa are where the
  thumb metacarpal points as seen from the WRIST joint. The thumb's own
  flexion axis is 48 to 92 degrees from the wrist joint's +x on those 69
  frames (median 77) and 65 to 89 from the palm's (median 76), so the pair
  is not the paper's and must not be compared with it; what exactly relates
  the two cannot be stated, because the paper gives its thumb reference
  frame as a drawing, not in numbers. Like the CMC, the pair includes the
  bend of the wrist; `*_palm_ref` gives it against the palm joint, with
  `forearm_to_palm_deg` beside it.
- PIP and DIP. The paper's are hinges. The tracker's INTERMEDIATE and
  DISTAL rows carry a sideways part as well (abd_deg up to 13 and 17
  degrees on that session); `*_PIP` and `*_DIP` are the flexion alone.
- Twist. The paper's finger bones do not turn about their own axis. The
  twist column is reported for every joint and is not one of the 24.

## The three commands

Every command runs from the repo folder:

```
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
```

### Camera only, bare hand: see the joint frames live

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --live
```

The window shows the IR picture with every tracked hand (both, when both
are in view) and a 1 cm triad at every joint (x red, y green, z blue) and,
on the right, one table per hand, left then right: the 26 rows (x y z in
mm, flexion, abduction) and the 24 angles, each from that hand's own
frames. One hand is drawn bright, the other a little dimmer; `--hand left`
or `--hand right` shows one hand only, with a single wider table. Keys:

| Key | What it does |
|---|---|
| h | makes the other hand the bright one in the picture |
| g | with `--glove`: switches the tables between the camera hands and the glove hands |
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
copies the CSVs and the PDF into the hand-in folder and never the PNGs.
The PDF is the drawing's hand-in form, and the only images a hand-in folder
may hold are the grasp set's infrared pictures cut to the hand
(`docs\grasp_recording.md`, step 10); the packager refuses any other. For
the grasp set it also writes the joint key beside them (above), and
`scripts\rebuild_grasp_set.py` runs this step, the packaging and the check
in one command.

Each drawing shows which arrow is which: every axis is a 10 mm arrow with
its head at the positive end, +x red, +y green, +z blue, with a legend that
says what each means; each digit has its letter (T, I, M, R, L) beyond its
fingertip in both views; each joint has the paper's name for it, the five
at the base of the hand in a row under the wrist with a leader to each; the
wrist is marked as the origin; and each view says where the wrist frame's
axes point in it. The PDF is written without a creation time, so the same
pages give the same file.
