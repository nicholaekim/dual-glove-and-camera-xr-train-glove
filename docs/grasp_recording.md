# Recording the grasp set (Set A)

The runbook for Set A of the professor's protocol
(`docs/grasp_and_flexion_protocol_plan.md`): every grasp in
`protocols/grasps.json` held still over the Ultraleap camera with the bare
left hand, 3 takes of 5 seconds each, no object in the hand. The recorder is
`scripts/leap/record_poses.py --protocol`. What it writes is fixed by
`docs/protocol_formats.md` (sections 1, 3, 5 and 7):

    recordings\protocol\grasps\<YYYYMMDD_HHMMSS>_left\
      session.json                     the whole session, rewritten after every take
      leap\<take>.jsonl                every camera frame of the take
      stills\<take>.png                one IR picture of the hand, cropped to the hand
      keypoints\<take>_keypoints.txt   the take's summary frame, professor's format
      meta\<take>.json                 the numbers and your decision
      rejected\                        takes not kept, same layout, plus <take>.reason.txt

A take is named `<grasp>_left_take<N>_<YYYYMMDD_HHMMSS>`, for example
`hook_left_take2_20260928_141503`.

## What happens in one take

1. The console names the grasp, describes its shape and repeats the
   orientation envelope (below).
2. A 5 second countdown, with a beep on each of the last three seconds. Get
   the hand into the grasp during the countdown.
3. A high beep: recording starts. Hold the grasp still for 5 seconds.
4. A low beep: recording stops.
5. The recorder checks the take against the acquisition gate (below). A take
   that fails is moved to `rejected\` and recorded again on its own.
6. A take that passes shows the review in the COPY THIS window: the still
   of your hand and the numbers. You decide with one key.

## The COPY THIS window

For each grasp, the picture to copy, cut from the paper's figure, appears
large in a window titled COPY THIS, with the grasp's name, the take number
and the countdown, then HOLD STILL and the seconds left while the take
records. After the take the same window shows the review, the paper's
picture beside your still, and then the next grasp's picture. The sheet
`grasp sheet.pdf` beside the papers is only a backup. If a picture is
missing, the window shows the grasp's shape in words instead and the
console says so.

## The orientation envelope

Every grasp is recorded inside this envelope (plan D3):

- the hand is 20 to 40 cm above the module;
- the wrist is inside the camera's view;
- every finger chain is visible to the lens;
- no finger is edge-on to the lens;
- a grasp that is naturally palm down (lateral key pinch, hook, extension
  plate) is turned only as far as the four points above need.

You do not have to write the rotation down. For every take the recorder
measures the palm's height and its angle to the lens (0 degrees is the palm
square to the lens, 90 is edge-on) and writes both into the take's meta
file.

## The acquisition gate

Checked automatically after every take:

- at least 90 % of the take's frames tracked;
- the hand not lost inside the static interval. The static interval is the
  2 seconds of the take in which the hand moved least; the summary frame is
  chosen from it. Lost means gone from the data for more than 0.1 s before
  the tracker finds it again. The tracker also re-labels a hand it never
  stopped seeing (a new hand id with no hole in the data; seven of nine
  "losses" in the 60 s test of 2026-09-30 were that); those are counted in
  the review line and the meta but do not reject the take.

The gate is the minimum for a usable take, not the decision. That is yours,
at the review.

## The review window

It shows the still, the grasp's name and these numbers:

- tracked %: the share of the take's frames in which the hand was tracked;
- grab and pinch: the tracker's own 0 to 1 values for the summary frame;
- curls: one number per finger, the same curl every other report in this
  repo uses;
- palm height and angle to the lens: check the height is 20 to 40 cm.

Compare the still with the paper's figure, fingertip by fingertip. A take
can be 95 % tracked with one fingertip wrong. Then press one key:

| Key | What it does |
|-----|--------------|
| Enter or space | keeps the take |
| r | records the take again; this attempt goes to `rejected\` with the reason "operator redo" |
| q | ends the session; this attempt goes to `rejected\` |

If a key does nothing, click the COPY THIS window once and press it again. The
key and the time you pressed it are written to the meta file and to
session.json.

## The camera window

The live IR picture with the tracked skeleton drawn on it. Use it to put
your hand inside the envelope during the countdown. It has a LEFT line and a
RIGHT line: that label is the tracker's guess, and it often calls the left
hand "right". Ignore the label. What matters is that one line says tracked
and its height reads OK. The other line saying NO HAND is expected.

## When a take is rejected

The console prints `REJECTED by the gate:` and the reason, which names the
losses behind it, for example `tracked 72 percent: lost 3 times, longest
1.4 s with the hand at 49 cm (too high: keep the palm 25 to 35 cm above the
module); the gate needs 90 percent`. Every loss is also listed in the take's
meta file under `gate.losses`, and `docs\tracking_quality.md` explains each
cause. The attempt's files move to `rejected\` beside a `<take>.reason.txt`, and the
recorder counts down to the same take again, up to 2 more times. Before the
next countdown ends, fix the cause:

- tracked under 90 %: bring the hand inside 20 to 40 cm, keep the wrist in
  view, and turn the hand so no finger is edge-on;
- the hand was lost inside the static interval: hold the grasp still from
  the high beep to the low beep and keep the hand over the middle of the
  module, 25 to 35 cm up.

After 3 failed attempts the recorder moves on to the next take. The end
table then lists that grasp as short of its takes, and step 7 below records
what is missing.

## Steps

### Step 1. No hardware: open PowerShell in the repo folder

```
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
```

Every command below runs from this folder.

### Step 2. No hardware: rehearse the whole session on synthetic hands

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --mock --protocol protocols\grasps.json --hand left --auto-accept --takes 1 --duration 1 --prep 0.5 --no-open
```

It takes about 30 seconds and ends with a table in which every grasp says
yes. The rehearsal is written to `recordings\protocol_mock\grasps\`, never
beside the real sessions, and its stills say MOCK.

### Step 3. Camera only, no hand over it: check the camera

```
.venv\Scripts\python.exe scripts\leap\check_setup.py
```

No line may say FAIL. Line 7 says WARN while no hand is over the camera,
which is fine. If a line says FAIL, do what that line says, then run this
step again.

### Step 4. Camera only, bare left hand: see the joint frames live

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --live
```

Hold the left hand over the module. The window draws a small x y z triad
(x red, y green, z blue, 1 cm long) at each of the 26 joints and lists, on
the right, every joint's position in millimetres from the wrist and its
flexion and abduction, then the paper's 24 angles (Cobos et al. 2009). Close
a fist slowly: the PIP numbers climb toward 90. Press n to highlight one
finger at a time, s to save a picture of the window into
`recordings\joint_frames\`, q to close it. What the numbers mean:
`docs\joint_frames.md`. The window only reads the camera, so it can also
stay open while you record.

### Step 5. Camera only, bare left hand: run the 60 second tracking test

Go through the physical checklist at the top of `docs\tracking_quality.md`
first (module flat, lenses up and wiped, cable straight into the laptop, no
sunlight, one hand over the module). Then:

```
.venv\Scripts\python.exe scripts\leap\tracking_quality.py --hand left
```

Follow the instructions it prints and shows in the camera window for one
minute, slowly. It ends with a line that starts with `VERDICT:`. If the
verdict names a cause, apply its fix and run this step again. Go on to the
next step when the verdict says no losses, or when the only losses are the
ones you caused on purpose (50 cm, the far side, the palm turned away).

### Step 6. Camera only, bare left hand: record the grasps

Before you press Enter: the module flat on the table with the lenses up, no
sunlight on it, and your right hand resting on the table away from the
module. The picture of each grasp appears in the COPY THIS window.

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --protocol protocols\grasps.json --hand left --operator "N Kim"
```

It waits until it sees a hand, then runs every grasp in the file, 3 takes
each. Plan on about 15 minutes of recording plus your time at the reviews.
At the end it prints the table (grasp, take, accepted, tracked %, grab,
pinch) and the folder, and opens the folder in Explorer.

### Step 7. Camera only, bare left hand: record the takes the table says are missing

Do this step only when a line under the table starts with
`Short of 3 kept takes`. Below that line, after "Record the missing takes
into this same session:", the recorder prints the exact command, which looks
like this:

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --protocol protocols\grasps.json --hand left --resume "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\recordings\protocol\grasps\20260928_140501_left"
```

Copy the printed command, not this example. It adds the missing takes to
the same session folder and carries the take numbers on, so Set A stays one
session (plan D9).

### Step 8. No hardware: look at the session folder

```
explorer recordings\protocol\grasps
```

Open the newest folder. Every grasp should have 3 files in `leap\`,
`stills\`, `keypoints\` and `meta\`. The stills stay on this machine: they
are for our own checking and are not handed in.

### Step 9. No hardware: write the joint frames of the session

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --session recordings\protocol\grasps\20260928_140501_left
```

Put your session's folder name in place of `20260928_140501_left`: it is
the newest folder you opened in step 8. For every kept take the script
writes the summary frame's 26 joints (position, axes, angles) to
`joint_frames\<take>.csv` inside the session folder, a drawing of it to
`joint_frames\<take>.png`, and one `joint_frames\joint_frames.pdf` with a
page per take. The packager later copies the CSVs and the PDF into the
hand-in folder, never the PNGs.
