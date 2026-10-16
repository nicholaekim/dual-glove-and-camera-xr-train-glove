# Pilot: how much does the Leap lose when an object is in the hand, and does the hand-on-object hint get it back?

Written 2026-10-16. One session of recording, about an hour and a half with the gloves on, that produces the
three numbers any object-aware correction project has to be justified by: the camera's disagreement with the
glove for a mimed grasp, for the same grasp around a real object, and for the same grasp around the object with
Ultraleap's `hand_on_object` tracking requested. No model is built here. If the object makes little difference, or
the hint alone removes most of it, the project changes shape before any time is spent on it.

One limit, known before recording. Ultraleap's Tracking Models page (docs.ultraleap.com/hand-tracking/Hyperion/
trackingmodels.html, read 2026-10-16) lists Hand On Object with "Current Compatible Camera Positions: HMD only", and
the Hyperion download on that page is for the Leap Motion Controller 2. This pilot uses the Stereo IR 170 on the
desk, in desktop tracking mode. The service here accepts the hint (the license flag is enabled and the recorder
confirms the request), but accepting a request is not the same as applying the model to this camera in this
position, and nothing in the API reports which model is running. So condition C is EXPLORATORY: if C is no better
than B, that says nothing about Hand On Object itself, only that it did not help this unsupported placement; if C is
better, that is worth having and worth checking again in a supported head-mounted placement before it is relied on.
A separate experiment with the camera head-mounted (HMD mode) is the supported way to test the model; it is not
this pilot, because A and B must stay a desktop comparison to speak to the grasp set.

Every step below names its hardware setup. Do one step at a time.

## What is being measured, and against what

* The camera: the Ultraleap Stereo IR 170 through the tracking service, desktop mode, as in the grasp set.
* The reference: the StretchSense glove's finger flexion. The glove senses stretch on the back of each finger, so a
  finger hidden from the camera by an object is still measured. It is a reference, not a truth: its angles come from
  its own calibration (the pinch procedure of 5 October), it drifts with fit, and it does not measure abduction or the
  thumb's base joint usefully. So the comparison is flexion only, at the MCP and PIP joints of the index, middle, ring
  and little fingers (8 joints), on a still hand.
* The quantity: for each take, ONE common held window for both sensors, the last 3 s of the 5 s hold by the shared
  wall clock the two recorders stamp (never a window each sensor picks for itself: when the object unsettles the
  camera, its own stillest 2 s would pick a kinder interval than the glove's and flatter the camera). Inside that
  window, pair every camera frame with the nearest glove frame in time (as `scripts/fuse_poses.py` pairs), take
  the flexion angles from `xr_hand.joint_frames` on both, and the absolute difference per joint per pair. Per take:
  the median paired-frame difference per joint, and the camera's coverage of the window (frames with the hand over
  frames expected at its rate) and its losses. Per condition: the median and the largest of the per-take medians over
  the takes and the 8 joints, per-joint medians, and coverage reported SEPARATELY from angle error, because a method
  that drops the hardest frames looks accurate if the two are mixed. Each sensor's representative frame (the stillest
  2 s, as in the grasp set) is kept as a secondary summary and for pictures only. Both sensors' values are kept; a
  difference is not attributed to either.
* Why a still hand: the glove's delay is not settled. The measurement of 5 October 2026 (`xr trainer/measure_glove_lag.py`
  v3, rank correlation on repeated closes and opens, both gloves on the Bluetooth 4.0 dongle; traces
  `xr trainer/glove lag tests/20261005_114958_both_left.csv` and `_right.csv`, summary `lag_tests.csv`) gave the LEFT
  glove +470 ms on both edges over 13 closes and the right about 0 on only 4 slow closes, which does not formally
  stand. Earlier figures (the fusion profile's 0.10 s left / 0.47 s right, the 455 ms of 20 September) were read off
  single closing movements, which the 5 October work showed measures the glove's response curve, not its delay; the
  delay has also moved between gloves from session to session. On a held pose none of this matters, which is why this
  pilot uses held grasps and not movements, and why its window starts 2 s after the beep.

## Conditions

| Condition | Hand | Recorder option |
|---|---|---|
| A, mimed | the grasp in the air, no object, as in the grasp set | none |
| B, object | the same grasp around the real object | none |
| C, object + hint (exploratory, see above) | as B, with `--hint hand_on_object` | `--hint hand_on_object` |

Three grasps, chosen because they hide fingers in three different ways:

| Pose name given to the recorder | Grasp | Object class (one exact object, chosen and measured in step 2b) |
|---|---|---|
| `cylindrical_object` | cylindrical power grasp (p1_cylindrical) | one cylinder about 6 to 7 cm across (a specific bottle or tin) |
| `spherical_object` | spherical grasp (p1_spherical) | one ball (a specific tennis ball) |
| `tip_object` | tip pinch (p1_tip) | one pen |

One physical object per grasp, the same one in B1, C and B2; a second cylinder would be another uncontrolled variable.
Each gets an identifier (OBJ-CYL, OBJ-BALL, OBJ-PEN), its dimensions (diameter and length, mm) and mass (g), written
once in `recordings\pilot_hoo\objects.txt` and repeated in every object run's `--note`.

5 takes per grasp per condition: 45 recordings, each a 5 s hold, plus 3 for the drift check at the end. The glove's reading creeps with time on the hand
(the drift seen in the earlier hold tests), so the conditions are not run in one fixed block each: condition B is
split around C, B1 (3 takes per grasp) before C and B2 (2 takes per grasp) after it, and condition A is repeated at
the end (1 take per grasp). If B1 and B2 differ from each other as much as B differs from C, the B-versus-C
difference is time, not the hint. One output folder per run, because a folder holds one hint setting (the
recorder refuses an output folder that already holds takes made under a different hint setting). Order: A (5) -> B1 (3) -> C (5) -> B2 (2) -> A-check (1).

The recorder's pose check knows only its named poses, so for these three names it reports "unchecked": whether the
hand made the grasp is the operator's eye, as it was for the grasp set, and the paper's picture should be on screen.

## Steps

**Step 1. No hardware.** Confirm the software: in the repo, run the toggle's tests and the setup check.

    .venv\Scripts\python.exe -m pytest tests -q -k hint
    .venv\Scripts\python.exe scripts\leap\check_setup.py

The second command must say the service is running (it does not read the license flag). The recorder reads the
Hyperion flag itself every time it starts with `--hint`, and writes it to each take's metadata; it was read as enabled
on this machine on 2026-10-16 (service 6.2.0). The flag proves the feature is enabled on this installation, nothing
more. If the recorder refuses with "no Hyperion license", stop: condition C cannot be recorded on this machine and the
pilot is A and B only.

**Step 2. Camera only, bare hand.** A one-take dry run with the hint, to see the request confirmed before the gloves
go on:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --mock-glove --hand left --poses open_palm --takes 1 --hint hand_on_object --out-dir recordings\pilot_hoo\dry_run

Open the take's `.meta.json` in `recordings\pilot_hoo\dry_run\leap\`: `tracking_hints` must read
`["hand_on_object"]`, `hints_confirmed` true, `license_hyperion` true. If not, stop and fix before recording.

**Step 2b. No sensors (a ruler and a kitchen scale).** Choose the three objects, measure each (diameter, length,
mass) and write `recordings\pilot_hoo\objects.txt`: one line per object, identifier, what it is, dimensions, mass.
These three objects are the only ones used for the rest of the pilot.

**Step 3. Gloves on + camera.** Put the gloves on as the vendor advised (sleeve gussets to the webbing, wiggle and bend
for a few seconds, then calibrate in XR Trainer; a closed hand in the calibration is closed, not squeezed). Right hand
rests on the table for the whole pilot, outside the camera's view. Connect the LEFT glove only, if XR Trainer allows
a single glove; an unused second glove adds nothing and the glove stream's timing has varied between sessions. If
the software insists on both, connect both for every one of the five runs, never a mix, and say so in every
`--note` ("both gloves connected"). XR Trainer streaming to port 9002, no other glove client running. Place the
three measured objects within reach, and the paper's three pictures on screen.

**Step 4. Gloves on + camera.** Condition A, mimed, 15 takes:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 5 --out-dir recordings\pilot_hoo\A_mimed --note "condition A mimed, no object"

Each take is the coached protocol: open hand steady over the module, beep, form the grasp in the air, hold still for
the 5 s, open. Nothing in the hand.

**Step 5. Gloves on + camera.** Condition B1, object, 9 takes (3 per grasp):

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 3 --out-dir recordings\pilot_hoo\B1_object --note "condition B1 object: OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt; <left glove only | both gloves connected>"

Pick the object up with the right hand and place it into the open left hand after the tracker has acquired the open
hand, then close on it at the beep and hold. Put it down between takes, keeping the right hand out of the camera's
view. The objects are the three of objects.txt and no others.

**Step 6. Gloves on + camera.** Condition C, object with the hint, 15 takes, the same way as step 5:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 5 --hint hand_on_object --out-dir recordings\pilot_hoo\C_object_hint --note "condition C object with hand_on_object: OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt; <glove setup as before>"

**Step 7. Gloves on + camera.** Condition B2, object, 6 takes (2 per grasp), the same way as step 5:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 2 --out-dir recordings\pilot_hoo\B2_object --note "condition B2 object after C: OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt; <glove setup as before>"

**Step 8. Gloves on + camera.** Drift check, condition A again, 1 take per grasp:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 1 --out-dir recordings\pilot_hoo\A_mimed_again --note "condition A repeated at the end, drift check"

Take the gloves off. The hardware part is over.

**Step 9. No hardware.** Export both sensors' joint angles for every take of the five sessions:

    .venv\Scripts\python.exe scripts\joint_frames_view.py --session recordings\pilot_hoo\A_mimed --no-window

and the same for `B1_object`, `C_object_hint`, `B2_object`, `A_mimed_again`. The per-take tables (26 joints, flexion
and abduction, for the glove take and the camera take of the same name) are the secondary summary; the paired-frame
comparison reads the raw takes themselves.

**Step 10. No hardware.** The comparison itself is a small script to write once the data exists (it is not written
yet, on purpose: the files it reads should exist first). What it must do, so that nobody argues about it afterwards:
for each take, cut the common window (last 3 s of the hold, shared wall clock), pair camera frames to the nearest
glove frames, take the 8 flexion angles (index, middle, ring, little; MCP and PIP) from both through
`xr_hand.joint_frames`, form the absolute differences per pair, and report per take the per-joint medians and the
camera's coverage and losses over that window. The PRIMARY report is per grasp: for each of cylindrical, spherical
and tip, the five runs A, B1, C, B2 and A-check side by side, each with its per-joint medians, the median and largest
of its per-take medians over the 8 joints, and its coverage, kept apart from error. B1 and B2 stay separate columns;
a pooled B over the five takes may be added as a descriptive value only, because the bracketing exists to show
time-on-hand drift and pooling throws that away. A three-grasp pooled summary is secondary, so that one grasp's
failure cannot hide inside a condition's average. The glove's drift between step 4 and step 8 is reported on the
glove alone. Put the result beside the data as one CSV and one page of text.

## How the result is read, decided before recording

No numerical threshold is set in advance. The patterns are:

| Pattern | Reading |
|---|---|
| B worse than A | the object-held condition gives a larger camera-glove disagreement and/or lower camera coverage, consistent with occlusion by the object. A fall in coverage or a rise in frame loss is direct evidence that the camera's tracking degraded; angle disagreement alone cannot say which sensor moved (A and B are not the same hand geometry, and the object touches the glove too) |
| C better than B (and B1 and B2 alike) | the hint recovers something even in an unsupported placement; confirm in HMD mode before relying on it |
| B worse than A, C about equal to B | the strongest case for an object-aware correction on the desktop; it does not show that Hand On Object fails, which this placement cannot test |
| B about equal to A | little evidence that this object condition needs correcting |
| B1 and B2 differ as much as B and C | the B-versus-C difference is time on the hand, not the hint |

Coverage is read beside error, never folded into it: a condition that loses the hardest frames and reports a low
error on the rest has not done better. The medians and the coverage go to the professor with these readings.

## What would make the pilot invalid

* A take whose meta shows `hints_confirmed` false in condition C, or true in A or B: the condition was not what its
  folder says. Drop it and record again.
* The glove recalibrated between conditions. If it has to be recalibrated, start the five sessions again.
* An object that hides the glove's back-of-finger sensors from nothing but the camera is the point; an object that
  presses the sensors (a tight grip on a hard edge) changes the glove reading itself. Hold the objects, do not squeeze.
* The right hand in the camera's view. Keep it on the table.

## What this pilot does not do

It does not test Hand On Object in the camera position Ultraleap supports for it (head-mounted); that is a separate
experiment. It does not build or train anything, does not re-project anything onto images, and does not decide what "good
enough" is. It also cannot tell whether a camera-versus-glove difference is the camera's error or the glove's: the two
disagree on mimed grasps too, which is exactly why condition A is recorded alongside B and C.
