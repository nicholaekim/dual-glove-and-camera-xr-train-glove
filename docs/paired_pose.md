# Paired pose comparison (glove + Ultraleap, nearest the photo's moment, one frame)

2026-11-03, runbook rewritten 2026-11-04 for the ten-pose session. What the professor asked
for (her request), per pose: a photo of the hand, the Leap hand model and the glove hand
model, "all showing the same moment", and the xyz of the same 21 hand points from both
sensors in the same units, origin and axis directions, with where the two disagree.
`scripts/paired_pose_compare.py` makes that from one session recorded by
`scripts/record_simultaneous.py --camera leap`. Coordinates only, no joint angles. The
`README.md` it writes into its output folder is the hand-in text; this page is how to
record and run it.

## The ten poses (left hand, approved by the professor)

open_palm, fist, index_point, thumbs_up, peace, pinch, finger_spread, thumb_opposition,
crossed_fingers, bent_wrist.

- The first six have a pose classifier (`leap_hand.pose_check`). The other four record
  "unchecked": "unchecked" means only that no pose classifier exists for them. They get
  every other check exactly like the six: timing, stillness, handedness, status and the
  visual check.
- finger_spread, thumb_opposition, crossed_fingers and bent_wrist are challenging for the
  glove's fixed-template, flexion-only model. Their measured distances are reported next
  to that model constraint; they are not stated in advance as large, and they are not
  attributed to either sensor.
- bent_wrist is a CONTROL. The wrist-centred, independently palm-aligned frame removes the
  hand's orientation, so a bent wrist with an unchanged hand shape cannot show as a
  landmark distance. bent_wrist is recorded to demonstrate what this comparison cannot
  measure, and kept because the professor approved the list; its distances must not be
  read as wrist-angle disagreement. The results README prints that sentence on the
  bent_wrist rows, and every bent_wrist take's notes.txt carries it.

## Rules for the whole session

- One calibration in XR Trainer at the start of the session. No recalibration between
  parts. If one is unavoidable, run the launcher with the reason in quotes after its name
  (it goes into every take's `--note`) and write it in the results README.
- Right hand resting on the table, outside the camera's view.
- Left hand 20 to 28 cm above the module.
- Every take starts from an open palm.
- Nothing is ever deleted. A take that fails a check stays on disk under its own take
  number; re-running a launcher records the NEXT take number (`--first-take auto`: one
  more than the highest take number of that pose already in the folder, kept or
  rejected), never overwrites.
- The acceptance checks apply to EVERY take, not only the first (see "Take selection").
- Take numbering: a clean session is 20 takes, two per pose. A session with repeated
  setup-gate attempts holds more: open_palm take 3 and beyond are normal there, and the
  final open-palm recording in part 2 never overwrites an earlier take
  (`--first-take auto`). The selection walks every take of a pose present in
  `comparison.csv` in ascending take number; it never assumes takes 1 and 2 are the only
  ones.

## Recording gate (before any hardware capture)

- [x] Full test suite result, filled in by the planning session before hardware capture:
  `.venv\Scripts\python.exe -m pytest tests -q -x` -> 1572 passed, 5 skipped, exit 0, 2026-11-04 (on the final code, after the last edit of this round). Method review: VERIFIED by the independent reviewer the same day. After the session, on the final code (2026-11-06): 2 failed, 1572 passed, 5 skipped under machine load; both failures are paced mock-recording tests unrelated to this work and pass alone (2 passed in 59.52 s).
- [x] Real-camera dry run passed (runbook step 1): 2026-11-05, take
  open_palm_left_take1_20261105_231948 (camera only, bare left hand, mock glove),
  moment_source "still capture time", sidecar capture_time_source
  leapc_image_timestamp, camera frame +1.6 ms from the photo, still, status ok. Repeated
  the same evening after the window change (still with nothing drawn on it): take
  open_palm_left_take1_20261105_233242, same moment source, camera frame -1.6 ms, the
  cropped still and its full frame with no coloured pixel and nothing drawn by eye.
- [x] First-take gate passed (runbook steps 3 to 9): 2026-11-05 23:42, take
  open_palm_left_take1_20261105_234207, all five checks passed, answered y; the session
  (part 2) followed at once: 20 takes, 5 recorder-refused attempts kept under rejected/,
  selection 10 of 10 poses accepted (pinch by take 2, take 1 "moving (both)"), outputs OK.
- [x] Independent review of the session and the reporting corrections: VERIFIED 2026-11-06; operator approval of the filled sheet recorded 2026-11-06T00:09:19-05:00.

A passing suite does not replace the real-camera dry run or the first-take gate: the
suite runs without the camera and the gloves.

## Runbook

Run from the repo root; the launchers are in `launchers\`.

1. Camera only, bare left hand, no gloves: double-click `sidecar_dry_run.cmd` (on the
   Desktop). Its scope: it validates the camera's timestamp-sidecar path only, not
   two-sensor timing, calibration or tracking quality. PASS when its `moment_source` says
   "still capture time".

2. Gloves on, camera not needed: calibrate both gloves once in XR Trainer. This is the only
   calibration of the session.

3. Gloves on + camera, left hand: double-click `launchers\record_ten_poses_1_first_take.cmd`.
   It records one open_palm take, runs the tool and prints `comparison.csv` and the five
   checks.

4. Gloves on + camera: on the newest open_palm row, check `moment_source` is "still capture
   time".

5. Gloves on + camera: check `cam_dt_ms` and `glove_dt_ms` are both within -50..50.

6. Gloves on + camera: check `stillness` is "still".

7. Gloves on + camera: check `handedness` is "same".

8. Gloves on + camera: open that take's `figure.png` and confirm by eye that the photo and
   both models show an open left palm.

9. Gloves on + camera: answer the launcher's question "Did all five checks pass? (y/n)".
   On n it asks "Reason:" and records it with
   `paired_pose_compare.py --gate-fail results\paired_pose\ten_poses_left --newest-of open_palm --reason "<text>"`
   (one row in `setup_gate.csv` for the open_palm take with the highest take number).
   Then fix the setup and repeat from step 3. The failed take stays; the re-run records
   the next take number; the take selection rejects the failed take with
   "setup gate failed: <reason>".

10. Gloves on + camera, left hand: double-click `launchers\record_ten_poses_2_rest.cmd`.
    It records round 1 (take 1 of each), round 2 (take 2 of each) in one interleaved order,
    the same in both rounds, so the four glove-challenging poses are spread among the six:
    finger_spread, fist, thumb_opposition, index_point, crossed_fingers, thumbs_up,
    bent_wrist, peace, pinch. Then open_palm's second take (the part-1 take counts as
    open_palm's first when it passed). Then the tool on the whole session. If it stops
    partway, run it again: every pose continues at its next take number.

11. No hardware: open `results\paired_pose\ten_poses_left\visual_checks.csv`. For every take,
    look at its `figure.png` and write yes or no in `intended_pose_seen`. Yes: the photo
    shows the intended pose being held (not a transition, not another pose) and each model
    shows that same hand as far as its sensor can represent it. A sensor's known inability
    is the disagreement under study, not a no: the glove reads finger flexion only, with a
    template spread and no wrist, so its model cannot show a pinch closing, a true spread
    or crossed fingers, and no model shows the wrist angle of bent_wrist (the control). No:
    the photo shows a different pose or a hand in transition, or a model contradicts the
    photo beyond its sensor's limits (a glove reading an open hand while the photo shows a
    fist). This wording was fixed on 2026-11-05 before the first sheet was filled (G23): the
    earlier "did the photo and both models show the intended pose?" would have sent pinch
    and bent_wrist to re-record for what their sensors cannot show. For
    `ten_poses_left`: Visual acceptance criterion clarified after recording but before visual decisions were entered; all 20 takes were assessed using the clarified criterion.

12. No hardware: run this one command.

        .venv\Scripts\python.exe scripts\paired_pose_compare.py --select results\paired_pose\ten_poses_left

13. No hardware: check the output folder is complete (re-reads every file).

        .venv\Scripts\python.exe scripts\paired_pose_compare.py --check-outputs results\paired_pose\ten_poses_left

14. No hardware: write `docs\sessions\<session>.md` (camera orientation as observed,
    criterion chronology, decision provenance, selection and exclusions, how to read the
    numbers) and rerun the tool so the README includes it
    (`.venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\<session>`, then
    steps 12 and 13 again, which put back the selection table and re-check the folder).

Rehearsal with no hardware: the day-2 takes, `scripts\paired_pose_compare.py
recordings\sync_day2`. Mock recordings go to `recordings\ten_poses_left_mock`, beside the
real session, never in it (the tool refuses mock takes: no still).

## Take selection (`--select`)

`visual_checks.csv` columns: take, pose, figure, intended_pose_seen, decided_at, note.
Rows are keyed by the take stem, which already contains pose, hand, take number and
timestamp (`<pose>_<hand>_take<N>_<YYYYMMDD_HHMMSS>`). `figure` is the figure.png the
operator must look at (relative to the results folder); `intended_pose_seen` is the
operator's yes or no; `decided_at` may be left blank. The normal run writes the sheet when
it is absent and adds a blank row for every take it does not list yet; existing filled
rows are never changed.

`setup_gate.csv` columns: take, declared_at (ISO local time), reason, revalidated,
revalidation_reason, revalidated_at. `--gate-fail <results folder> --take <stem>` (or
`--newest-of <pose>`) `--reason "<text>"` appends one row with `revalidated` empty. Any
take listed there with `revalidated` other than yes is ineligible, with the reason
"setup gate failed: <reason>", even when every automated check and the visual check pass.
To revalidate, the operator writes yes in `revalidated` and fills `revalidation_reason`
(`--select` refuses a yes with an empty `revalidation_reason`); the take is then eligible
and `selection.csv` shows "revalidated: <reason>" in its row. The normal run never deletes
or rewrites `setup_gate.csv`.

`--select <results folder>` writes `selection.csv`, one row per pose: pose, hand,
accepted_take, rejected_takes, the reason per rejected take, auto_checks ("passed" or the
failed checks), visual_check ("yes", "no", or "not filled" for a take the tool refused or
the setup gate blocked), setup_gate, pose_classifier ("checked" for the six, "none" for the
other four), action, note. It also writes the selection table into the README.

The predefined rule, applied in take order (take 1 first, then take 2, ...): a take is
accepted when ALL hold:

- `moment_source` is "still capture time";
- |`cam_dt_ms`| <= 50 and |`glove_dt_ms`| <= 50;
- `stillness` is "still";
- `handedness` is "same" ("opposite" and "indeterminate" reject the take with
  "investigate before re-recording");
- `status` is "ok" (palm frames valid, landmarks finite);
- `visual_checks.csv` says yes;
- the take has no unrevalidated row in `setup_gate.csv`.

Every take of the pose in `comparison.csv` is walked in ascending take number, however
many there are. The first take that passes is accepted; later takes are never considered once one is
accepted. Nothing is chosen by disagreement magnitude: `median_mm` and `max_mm` are read
only to see that they are finite. A pose with no accepted take is listed with
`accepted_take` empty and "re-record". `--select` refuses, naming the takes, when
`visual_checks.csv` has a blank cell for a take it needs (every considered take the tool
processed). Nothing is deleted: rejected takes stay on disk and in `comparison.csv`. The
recorder's own rejected attempts (`_attemptN`) are not candidates.

When `docs/sessions/<results folder name>.md` exists, the normal run includes its text
verbatim in the README under "Operator's session notes", after the run summary.

## Stills and the uncropped frame

The camera window crops each take's still to the tracked hand (`<take>.jpg`) and keeps the
uncropped frame it was cut from beside it (`<take>.full.jpg`): the same frame object, so the
same moment, and the timestamp sidecar (`<take>.jpg.json`) lists both (`"full"`). The
recorder moves the full frame with the still (accepted rename and rejected path); the
meta's `still_full` names it (null when absent). The hand-in rule is unchanged: uncut
pictures never go into a professor folder. The full frame stays in the session folder only.

## Outputs

`results\paired_pose\<session folder name>\` (git-ignored; `--out` puts them elsewhere): one
folder per take with `figure.png`, `camera.txt`, `glove.txt`, `points_common_frame.csv`,
`bone_lengths.csv` and `notes.txt`, then `comparison.csv`, `figures.pdf`, `README.md`,
`visual_checks.csv` and, after `--select`, `selection.csv`. `--takes <stem> ...` runs only
those takes (then `comparison.csv`, and so the selection, covers only them);
`--max-dt-ms` (default 50) is how far a sensor's frame may be from the moment;
`--include-rejected` also runs the attempts under `rejected\`.

## Conventions

- What the numbers are: a wrist-centred, independently palm-aligned shape disagreement,
  from an arrival-time matched, held-pose comparison. Physical simultaneity of the two
  selected frames is not verified; they are the frames nearest the photo's time on each
  sensor's own arrival clock, taken during a held pose. (The camera's capture_time
  approximates image capture, the glove's is packet arrival: |dt| <= 50 ms does not
  establish one physical instant, and the stillness check reduces but does not remove
  that.)
  Each sensor's palm frame is built from its own landmarks, so orientation differences
  between the sensors are removed, not measured; the sensors are not physically
  registered (the glove reports no hand position or orientation: its WRIST and PALM never
  move, measured 2026-09-16 and re-checked on day 2).
- The moment is the photo's: the still the camera window saves at the midpoint of the
  REC hold. Sessions recorded from 2026-11-03 on carry its capture time in the take's
  meta (`still_capture_time`, from the viewer's `<still>.json`: the image frame's LeapC
  timestamp on the wall clock); older ones, day 2 included, fall back to the still file's
  write time (plus an unknown write delay), and every output says which. The still has
  nothing drawn on it (no skeleton, no aiming ring, no caption): it is the window's IR
  frame with its brightness lifted for the eye (x2.2 + 8), mirrored as the window shows
  it, cropped to the tracked hand. Stills recorded before the window change of 2026-11-05
  (every take in results/paired_pose/sync_day2, and recordings/sidecar_dry_run take 1,
  recorded earlier that day) carry the
  tracker's skeleton overlay, the aiming ring and the recorder's caption, because the
  window used to draw before copying; their figures were made from those files as they
  are.
- Each sensor contributes the frame of its own file nearest that moment on the capture
  clock, within 50 ms; a take without one is listed with the reason, not skipped. Both
  capture clocks are the PC wall clock, but the glove's is packet ARRIVAL, after its
  transmission delay (about 0.47 s left, about 0 right, 5 October 2026), not corrected.
- Millimetres; the wrist as origin on both sensors; x = (M - W)/|M - W|,
  z = x × (P - I)/|x × (P - I)|, y = z × x (W wrist 0, M middle MCP 9, P pinky MCP 17,
  I index MCP 5), exactly `cam_hand.fusion.palm_frame` / `palm_basis`, built on each
  sensor's own hand. A take whose x and P - I are under 10 degrees apart, or that lacks
  one of the four, is refused as "degenerate palm frame (...)". No fit, no scaling (the
  unscaled distances are the result), no lag correction.
- The text files keep each sensor's native frame and the professor's layout (point 0 is
  the model's Palm point, not compared); the CSV holds the common frame (point 0 WRIST).
- The glove's label names the hand; the tracker's is its guess. `handedness` is a
  diagnostic ("same", "opposite (suspected mirrored tracker fit)", "indeterminate (margin
  under 5 mm)") read from each hand's mean palm-normal coordinate; nothing is reflected.
- `stillness`: each sensor's mean 21-point speed within 0.25 s of its selected frame,
  "moving" above 10 mm/s (`leap_hand.transition.POSE_SETTLE_SPEED_MM_S`); the take is
  processed either way (and the take selection rejects it).
- Glove bone lengths are a template hand; the tracker's change with the pose. The README
  lists each landmark's source on each sensor.
- A number is a distance between the two sensors' estimates, a disagreement; neither
  sensor is the reference.
- The figure shows the still as saved. Day-2 stills are the whole camera frame and show
  the operator; stills from the current viewer are cropped to the hand.
