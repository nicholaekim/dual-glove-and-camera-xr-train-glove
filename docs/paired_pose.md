# Paired pose comparison (glove + Ultraleap, nearest the photo's moment, one frame)

2026-11-03. What the professor asked for (her request), per pose: a photo of the hand, the
Leap hand model and the glove hand model, "all showing the same moment", and the xyz of the same 21
hand points from both sensors in the same units, origin and axis directions, with where
the two disagree. `scripts/paired_pose_compare.py` makes that from one session recorded
by `scripts/record_simultaneous.py --camera leap`. Coordinates only, no joint angles.
The `README.md` it writes into its output folder is the hand-in text; this page is how
to run it.

## Commands

Run from the repo root.

1. Rehearsal on the day-2 takes. No hardware.

       .venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\sync_day2

2. First-take gate of a new session. Gloves on + camera for every sub-step; do not record
   the other poses until all five checks pass.

   2.1. Gloves on + camera: record the session's FIRST take through
        `scripts\record_simultaneous.py --camera leap`, and leave both sensors on.

   2.2. Gloves on + camera: in a second terminal, run this one command.

            .venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\<session folder>

   2.3. Gloves on + camera: in that take's printed line (or its row in `comparison.csv`),
        confirm `moment_source` is "still capture time".

   2.4. Gloves on + camera: confirm both `cam_dt_ms` and `glove_dt_ms` are 50 ms or less
        in size.

   2.5. Gloves on + camera: confirm `stillness` is "still".

   2.6. Gloves on + camera: confirm `handedness` is "same".

   2.7. Gloves on + camera: open that take's `figure.png` (in
        `results\paired_pose\<session folder>\<take>\`) and confirm by eye that the photo and
        both reconstructed models show the intended pose.

   2.8. Gloves on + camera: if any of 2.3 to 2.7 fails, correct the setup, record that take
        again, and repeat from 2.2. Only when all five pass, record the other poses.

3. The new session, after all of it is recorded. No hardware for this step.

       .venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\<session folder>

4. Check the output folder is complete (re-reads every file). No hardware.

       .venv\Scripts\python.exe scripts\paired_pose_compare.py --check-outputs results\paired_pose\<session folder>

Outputs go to `results\paired_pose\<session folder name>\` (git-ignored; `--out` puts
them elsewhere): one folder per take with `figure.png`, `camera.txt`, `glove.txt`,
`points_common_frame.csv`, `bone_lengths.csv` and `notes.txt`, then `comparison.csv`,
`figures.pdf` and `README.md` for the session. `--takes <stem> ...` runs only those
takes; `--max-dt-ms` (default 50) is how far a sensor's frame may be from the moment;
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
  write time (plus an unknown write delay), and every output says which.
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
  processed either way.
- Glove bone lengths are a template hand; the tracker's change with the pose. The README
  lists each landmark's source on each sensor.
- A number is a distance between the two sensors' estimates, a disagreement; neither
  sensor is the reference.
- The figure shows the still as saved. Day-2 stills are the whole camera frame and show
  the operator; stills from the current viewer are cropped to the hand.
