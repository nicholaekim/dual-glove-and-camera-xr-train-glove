# Pilot: how much does the Leap lose when an object is in the hand, and does the hand-on-object hint get it back?

Written 2026-10-16; step 9 (the comparison script) written and pre-registered 2026-10-21, before any pilot data.
One session of recording, about an hour and a half with the gloves on, that produces the
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
* The quantity: for each take, ONE common held window for both sensors, the last 3 s of the 5 s hold on the clock
  the two recorders share (`capture_time`, the sensor's own capture instant, when every frame of both files has it;
  else `wall_time`; both are `time.time()`), never a window each sensor picks for itself: when the object unsettles
  the camera, its own stillest 2 s would pick a kinder interval than the glove's and flatter the camera. The window
  ends at the recorder's commanded end of the hold, `hold_end` in the take's meta (the recorder stops both sensors
  together, so a sensor that lost the hand before the end shows up as missing coverage, not as a shorter window);
  a take recorded before that key existed ends at the later of the two sensors' last stamps, and the report counts
  such takes. Inside that window, pair every camera frame of the hand with the nearest glove frame in time, within
  50 ms, with no lag shift (the hand is held still, see below); camera frames with no glove frame that close are
  counted as unpaired, and the paired fraction is reported. Take the flexion angles from `xr_hand.joint_frames`
  (wrist reference, flexion positive toward the palm) on both; per pair the DISAGREEMENT camera minus glove, wrapped
  to [-180, 180), signed and absolute. It is a disagreement, not an error: the glove is a reference, not truth. Per
  take: the median absolute and the median signed disagreement per joint, each sensor's own median per joint, each
  sensor's within-window stability per joint (the median absolute deviation, MAD, of its own angle over the paired
  frames, so a hand that adjusted during the window is visible rather than excluded), and each sensor's coverage of
  the window and its losses by the recorder's own definitions (`leap_hand.protocol.coverage`: the fraction of the
  window not inside a hole longer than 0.25 s; and the count of gaps over 100 ms). Per run and grasp: the PRIMARY
  quantity is the median over all take x joint values of the per-take median absolute disagreement; beside it the
  largest of those values, the per-joint medians (absolute and signed), the take counts (valid, expected, invalid,
  unusable), the paired fraction, the worst gap and the stability, and coverage reported SEPARATELY from
  disagreement, because a method that drops the hardest frames looks accurate if the two are mixed. No
  representative frame is computed; the picture of a take is the still the recorder saves at the midpoint of the
  hold (`stills/<take>.jpg`), which cannot be picked for a moment the camera looked best. Both sensors' values are
  kept; a difference is not attributed to either.
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
mass) and write `recordings\pilot_hoo\objects.txt`: one line per object, five fields separated by `|`, identifier,
what it is, diameter in mm, length in mm, mass in g (or the words `not measured`); `#` lines are comments:

    OBJ-CYL | <what it is> | <diameter_mm> | <length_mm> | <mass_g>
    OBJ-BALL | tennis ball | <diameter_mm> | <diameter_mm> | <mass_g>
    OBJ-PEN | <what it is> | <diameter_mm> | <length_mm> | <mass_g>

The object runs (steps 5 to 7) refuse to start until this file exists with all three lines. These three objects are
the only ones used for the rest of the pilot.

From step 3 on, every step is one Desktop launcher (copies in `launchers\pilot\`), each running
`scripts\pilot_run.py <run>`. It runs the command printed under the step, but one grasp at a time
(`--poses <grasp>`), so that the object for the grasp is announced before its takes (`USE OBJ-CYL: <its line>`)
and written into every take's `--note`. Before anything is recorded it checks, and prints every unmet check:
the glove setup, ONE for the whole pilot (`recordings\pilot_hoo\glove_setup.txt`, written once in step 3; a
different setup later is refused); for the object runs, `recordings\pilot_hoo\objects.txt`, which is frozen when
B1 starts (`objects.lock`: a changed file refuses C and B2); and the order. A run is COMPLETE when every grasp has
its expected takes (A 5, B1 3, C 5, B2 2, A-check 1 per grasp); a run has STARTED once its folder holds any
attempt, kept or rejected, or its label is in `runs_started.txt`. A run may start only when the previous run is
complete and no later run has started; a run that is short of takes is topped up for the missing takes only, with
the numbering continued (`--first-take`); a complete run is refused, so no take beyond the protocol is added. The
grasps of a run are recorded in order, and if one ends short (the recorder was quit with q, or lost takes), the
launcher stops there instead of moving to the next grasp. If the recorder itself prints a "redo" command at the end
of a run, do not type it (it has no `--first-take`): double-click the same step's launcher again, which records only
what is missing. In the commands below `<glove setup>` is `left glove only` or `both gloves connected`, as written in step 3; the
launcher fills it in. Nothing is hidden: the printed commands are the ones that run.

**Step 3. Gloves on + camera.** Put the gloves on as the vendor advised (sleeve gussets to the webbing, wiggle and bend
for a few seconds, then calibrate in XR Trainer; a closed hand in the calibration is closed, not squeezed). Right hand
rests on the table for the whole pilot, outside the camera's view. Connect the LEFT glove only, if XR Trainer allows
a single glove; an unused second glove adds nothing and the glove stream's timing has varied between sessions. If
the software insists on both, connect both for every one of the five runs, never a mix, and say so in every
`--note` ("both gloves connected"). XR Trainer streaming to port 9002, no other glove client running. Place the
three measured objects within reach, and the paper's three pictures on screen. Then double-click
`pilot_step3_gloves_left_only.cmd` (or `pilot_step3_gloves_both.cmd` if the software insisted on both): it writes the
glove setup once (`pilot_run.py A --gloves left --print`) and shows the step 4 command without recording.

**Step 4. Gloves on + camera.** Condition A, mimed, 15 takes:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 5 --out-dir recordings\pilot_hoo\A_mimed --note "condition A mimed, no object; <glove setup>"

Launcher: `pilot_step4_A_mimed.cmd` (`pilot_run.py A`).

Each take is the coached protocol: open hand steady over the module, beep, form the grasp in the air, hold still for
the 5 s, open. Nothing in the hand.

**Step 5. Gloves on + camera.** Condition B1, object, 9 takes (3 per grasp):

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 3 --keep-all-tracking --out-dir recordings\pilot_hoo\B1_object --note "condition B1 object: OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt; <glove setup>"

Launcher: `pilot_step5_B1_object.cmd` (`pilot_run.py B1`).

Pick the object up with the right hand and place it into the open left hand after the tracker has acquired the open
hand, then close on it at the beep and hold. Put it down between takes, keeping the right hand out of the camera's
view. The objects are the three of objects.txt and no others.

**Step 6. Gloves on + camera.** Condition C, object with the hint, 15 takes, the same way as step 5:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 5 --keep-all-tracking --hint hand_on_object --out-dir recordings\pilot_hoo\C_object_hint --note "condition C object with hand_on_object: OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt; <glove setup>"

Launcher: `pilot_step6_C_object_hint.cmd` (`pilot_run.py C`).

**Step 7. Gloves on + camera.** Condition B2, object, 6 takes (2 per grasp), the same way as step 5:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 2 --keep-all-tracking --out-dir recordings\pilot_hoo\B2_object --note "condition B2 object after C: OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt; <glove setup>"

Launcher: `pilot_step7_B2_object.cmd` (`pilot_run.py B2`).

**Step 8. Gloves on + camera.** Drift check, condition A again, 1 take per grasp:

    .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses cylindrical_object,spherical_object,tip_object --takes 1 --out-dir recordings\pilot_hoo\A_mimed_again --note "condition A repeated at the end, drift check; <glove setup>"

Launcher: `pilot_step8_A_mimed_again.cmd` (`pilot_run.py A2`).

Take the gloves off. The hardware part is over.

**Step 9. No hardware.** The comparison. The script exists before the data does: `scripts/pilot_compare.py`, written
on 2026-10-21, when `recordings\pilot_hoo` held nothing but the dry run of 19 October (`GATES_pilot_compare.md`,
gate P4), so it cannot have been tuned to the result. Its behaviour is pinned by `tests/test_pilot_compare.py` on
synthetic takes in the recorder's own file layout. Run it once, from the repo root:

    .venv\Scripts\python.exe scripts\pilot_compare.py --root recordings\pilot_hoo

Launcher: `pilot_step9_compare.cmd` (`pilot_run.py compare`; no hardware). The launcher refuses to run the
comparison while A-check is incomplete, so a half-collected pilot cannot produce what looks like the final report
by accident; a pilot that cannot be finished is closed on purpose with `pilot_run.py close --reason "<why>"`,
which writes `pilot_closed.txt` (time, per-run counts, reason) and then lets the comparison run with its missing
rows showing. The script itself, `pilot_compare.py --root recordings\pilot_hoo`, stays usable at any time for a
look at partial data.

It reads the five run folders by name (`A_mimed`, `B1_object`, `C_object_hint`, `B2_object`, `A_mimed_again`), every
accepted take's `.meta.json`, camera file and glove file, and writes three files into `recordings\pilot_hoo\comparison`:

* `takes.csv`, one row per accepted take: run, condition, grasp, take, note, `hints_confirmed`, status, the clock
  used, the window and where its end came from (`meta hold_end` or `later last stamp`), frame counts, pairs and
  unpaired camera frames, each sensor's coverage and gaps, and per joint the median absolute disagreement
  (`absdiff_med_*`), the median signed disagreement camera minus glove (`signed_med_*`), each sensor's own median
  (`cam_med_*`, `glove_med_*`) and each sensor's stability (`cam_mad_*`, `glove_mad_*`). The per-sensor medians are
  the secondary summary (an earlier draft of this step exported per-take tables with `joint_frames_view.py
  --session`; that mode reads protocol sessions with a `session.json`, which `record_simultaneous.py` never writes,
  so it was replaced by these columns).
* `summary.csv`, over the valid takes only: one row per grasp and run (`run_grasp`), a pooled-B row per grasp
  (`run_grasp_pooledB`, B1 and B2 together, descriptive only), one row per run over all grasps (`run_all`, secondary) with its own pooled B
  (`run_all_pooledB`), each with the primary quantity `absdiff_median`, `absdiff_largest`, per-joint `absdiff_*` and `signed_*`, the take
  counts (`n_valid_takes`, `n_expected_takes` by this protocol, `n_invalid_takes`, `n_unusable_takes`,
  `n_missing_takes`), the ATTEMPT counts read from `leap/` and `rejected/leap/` together (`n_attempts`,
  `n_rejected`, split into hand lost / re-acquired / low coverage / other by the recorder's own reason, and the
  recorder's coverage over all attempts, median and min: the recorder keeps a take only when the hand stayed
  tracked, so a camera that loses the hand with the object in it shows up here as rejected attempts, not as low
  coverage inside kept takes), paired fraction, worst gaps, stability and coverage, with a row for every run x
  grasp of the fixed grid whether or not it has takes; per grasp the glove's drift between step 4 and step 8 on the glove
  alone (`glove_drift_A2_minus_A`: the glove's median flexion in A-check, one take per grasp, minus its median in A,
  per joint, signed); and the pre-stated contrasts on the primary quantity (`contrast` rows, per grasp and pooled:
  `Bpooled_minus_A`, `C_minus_Bpooled`, `B2_minus_B1`, `A2_minus_A`, `abs_B2_minus_B1_minus_abs_C_minus_Bpooled`).
* `report.txt`: the PRIMARY report, per grasp, with the five runs A, B1, C, B2 and A-check side by side and the pooled
  B beside them: the take counts, per-joint absolute and signed medians, the primary quantity and the largest value,
  paired fraction, each sensor's coverage, gaps and stability, kept apart from disagreement, then the contrasts. B1
  and B2 stay separate columns because the bracketing exists to show time-on-hand drift and pooling throws that away.
  The three-grasp pooled block is secondary, so that one grasp's failure cannot hide inside a condition's average. The
  report prints numbers only; the reading is the table in the next section.

A take whose `hints_confirmed` contradicts its run (false in C, true in A or B) is listed under "invalid takes" with
its numbers kept in `takes.csv`, and left out of every summary; record it again. A run folder that is missing or has
no accepted take is reported as absent, so the script also runs on a partial session.

## How the result is read, decided before recording

No numerical threshold is set in advance. Every pattern below is read on ONE number, the primary quantity of
`summary.csv` (`absdiff_median`: the median over take x joint of the per-take median absolute disagreement, per
grasp), through the contrasts the script prints: "B worse than A" is `Bpooled_minus_A` positive, "C better than B"
is `C_minus_Bpooled` negative, each with its size stated beside the `largest` and stability rows. The pilot makes
NO categorical "about equal" or "as much as" claim: no equivalence margin is fixed in advance (nothing is known yet
about the camera-glove disagreement on a mimed grasp, so any margin chosen now would be a guess), and the script
assigns no pattern. For the rows below that speak of "about equal" and "differ as much as", the report gives the
contrasts (`B2_minus_B1`, `abs_B2_minus_B1_minus_abs_C_minus_Bpooled`) and the spread, and the reader sees how far
from zero they are. Coverage, paired fraction and the gap counts are read beside these, never inside them. The
summary grid is fixed before recording: the three grasps named above, the five runs and the pooled B, with the
expected take counts of this protocol (A 15, B1 9, C 15, B2 6, A-check 3 in all); a run x grasp with no take still
has its row and its `n_missing_takes`, so a grasp that was never recorded cannot disappear from the table. Per grasp
first; the pooled contrasts second.

| Pattern | Reading |
|---|---|
| B worse than A | the object-held condition gives a larger camera-glove disagreement and/or lower camera coverage, consistent with occlusion by the object. A fall in coverage or a rise in frame loss is direct evidence that the camera's tracking degraded; angle disagreement alone cannot say which sensor moved (A and B are not the same hand geometry, and the object touches the glove too) |
| C better than B (and B1 and B2 alike) | the hint recovers something even in an unsupported placement; confirm in HMD mode before relying on it |
| B worse than A, C about equal to B | the strongest case for an object-aware correction on the desktop; it does not show that Hand On Object fails, which this placement cannot test |
| B about equal to A | little evidence that this object condition needs correcting |
| B1 and B2 differ as much as B and C | the B-versus-C difference is time on the hand, not the hint |

Coverage is read beside error, never folded into it: a condition that loses the hardest frames and reports a low
error on the rest has not done better. The medians and the coverage go to the professor with these readings.

## Amendment of 2026-10-23: the object runs keep every attempt

Made during the first recording day, after run A was complete and before any object take was kept; reviewed in the
ChatGPT thread (agreed, with the four requirements below) before it was applied.

What happened. The first B1 attempt, the can in the gloved left hand over the desk-mounted IR 170, was tracked for
81 % of the 5 s hold and then lost for 0.5 s, so the coached recorder rejected it under its strict rule (coverage of
at least 0.90 and never lost for 0.5 s; the hold is cut short at the loss and the files go to `rejected\`). That
rule exists for the grasp set, where a lost hand means a bad take. Here it throws away exactly the frames the pilot
is about, and any object take it did keep would be one where the camera happened to cope: a selection toward
camera-friendly moments. The operator then recorded nine takes under the B1 label without any object ("I didn't use
objects at all, since the camera can't detect it and since it covers the glove"); they are kept as protocol
deviations in `B1_object\set_aside_no_object\` with a manifest (`protocol_valid` false, the reason, the statement,
the date), counted in no condition and never relabelled as mimed. The one object attempt stays where the recorder
put it, a pre-amendment rejected attempt, counted in the attempt audit and never as a kept take.

The rule from now on. The object runs B1, C and B2 are recorded with `--keep-all-tracking` (the launcher passes it;
the commands under steps 5 to 7 show it). With it: the open-hand acquisition and the settle stay strict, as before;
during the hold a camera loss, a re-acquisition (hand id change) or wrong-chirality frames are LOGGED and never used
to reject, and the hold always runs its full 5 s, so the glove is recorded for the whole hold whatever the camera
does; an attempt is kept unless the glove side failed or the hand was never acquired open. Runs A and A-check keep
the strict rule, which is how run A was recorded. Every meta, strict or not, now records `acceptance_policy`
(`strict` or `object_keep_all_tracking`), `protocol_amendment_date`, `hold_completed`, `cam_tracked_fraction` over
the commanded hold (the recorder's own hole-based coverage), `cam_max_gap_ms`, `cam_gaps_over_100ms`,
`cam_reacquisitions` and a `tracking_note` with the loss text that would have rejected the take. The comparison
carries them into `takes.csv` and, per run and grasp, the tracked fraction's median and min, the worst
re-acquisition count and the strict / keep-all take counts, beside the attempt rows: the professor can tell "the
angles agree when the hand is visible" from "the camera was hardly ever visible". No pre-registered quantity
changes: coverage, paired fraction and the attempt rows were always to be read beside the disagreement, never inside
it. Option rejected: coaching the hand position so the camera copes, which would tune the condition to the sensor
under test. The operator's second concern, that the object presses the glove, is the standing rule "hold, do not
squeeze", and the per-take stability (MAD) rows show adjustment.

Consequence for the recording day: the gloves came off while the recorder was changed. Because the glove's fit and
calibration must not change between conditions, the five runs are recorded again from step 3, run A included (15
takes, a few minutes). The first attempt's folder was renamed `recordings\pilot_hoo_2026-10-23_attempt1` (run A
valid, the B1 deviations, the pre-amendment attempt, its own objects.txt and glove_setup.txt) and the launchers
record the new session into a fresh `recordings\pilot_hoo`, so no launcher changes.

## What would make the pilot invalid

* A take whose meta shows `hints_confirmed` false in condition C, or true in A or B: the condition was not what its
  folder says. The comparison script lists it as invalid and leaves it out; record it again.
* The glove recalibrated between conditions. If it has to be recalibrated, start the five sessions again.
* An object that hides the glove's back-of-finger sensors from nothing but the camera is the point; an object that
  presses the sensors (a tight grip on a hard edge) changes the glove reading itself. Hold the objects, do not squeeze.
* The right hand in the camera's view. Keep it on the table.

## What this pilot does not do

It does not test Hand On Object in the camera position Ultraleap supports for it (head-mounted); that is a separate
experiment. It does not build or train anything, does not re-project anything onto images, and does not decide what "good
enough" is. It also cannot tell whether a camera-versus-glove difference is the camera's error or the glove's: the two
disagree on mimed grasps too, which is exactly why condition A is recorded alongside B and C.
