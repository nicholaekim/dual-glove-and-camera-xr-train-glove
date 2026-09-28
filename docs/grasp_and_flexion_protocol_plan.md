# Plan: the professor's grasp and finger-flexion recordings

Written 2026-09-27. Reviewed the same day in a new ChatGPT chat (the project
page would not load in Chrome, so the chat ran outside the project with the
context given in the prompt); the review outcome is in section 8 and its
accepted points are already folded in below.

Status 2026-09-27 evening: the tools in section 1 are built and pass their
mock end-to-end runs (record, check, package) for all three sets; nothing
has run on hardware yet, and the grasp list is still the placeholder.
Runbooks: `docs/grasp_recording.md` (Set A), `docs/protocol_recording.md`
(Sets B and C). File formats: `docs/protocol_formats.md`.

The professor asked for three sets:

| Set | What | Sensor | Hardware setup for the session |
|-----|------|--------|--------------------------------|
| A | Static grasp-type poses from the reference papers (spherical, cylindrical, lateral, pinch, tripod, and the rest shown in the papers). Final configuration only. | Camera ("Leap Motion" = the Ultraleap IR 170) | camera only, bare hand |
| B | Single-finger flexion, each finger separately: straight, slow bend to full flexion, hold about 1 s, slow straighten, 5 repetitions. Index also at a slow and a faster speed, several repetitions each. | Glove | gloves on + camera (camera as reference, see below) |
| C | Seven sequential movements (one finger at a time, reverse order, alternating, pairs, progressive closing, other configurations, flex and release in sequence). | Glove | gloves on + camera |

## 1. What is already in the repo and what is missing

Reuse without change:

- `scripts/leap/record_poses.py`: camera-only guided session with a custom
  `--poses` list, beeps, labelled JSONL in the glove's naming scheme
  (`<pose>_<hand>_take<N>_<stamp>.jsonl`, every frame carries `pose` and
  `take`), `--raw` keeps LeapC's own `.lmt`. This is Set A's recorder.
- `scripts/glove/export_prof_format.py`: writes the professor's 21-landmark
  text format from any JSONL folder (glove or camera). Used for every set.
- `scripts/leap/record_frame.py`'s medoid rule (the real frame closest to the
  take's mean after rigid alignment) for the one-frame summary of each grasp.
- `scripts/leap/finger_sweep.py`: paced bend/straighten cycles for one finger
  with the camera as reference. Its analysis (`leap_hand/diagnostics.py`:
  transfer curve, rail saturation, hysteresis, lag) is reused for Set B's
  report. Its recorder is not reused: it saves per-finger curls as CSV, not
  the 26-joint glove frames the professor's format needs, and it has no hold
  at full flexion.
- The coached recorder's parts (`scripts/record_simultaneous.py`): glove
  recorder at full rate with gap statistics, Ultraleap recorder with
  `capture_time` pairing, acquire gate (open palm over the module at 18 to
  28 cm), one hand-cropped IR still per take, `meta.json`.
- `scripts/fuse_live.py` warm-up (open 3 s, fist 3 s) to get each finger's
  open and fist endpoints for the session, so Set B and C curls can be read
  as fractions of that hand's range.

Missing (to build after this plan is approved):

1. `protocols/` folder with two JSON files: `finger_flexion.json` (five
   fingers, 5 cycles each, plus index slow and index fast) and
   `sequences.json` (the seven sequences as ordered steps, each step = the
   set of fingers that should be flexed). Sequences are data, not code, so
   the professor's "try other orders" is a new JSON entry.
2. `scripts/record_protocol.py`: cue-driven recorder. Reads a protocol file,
   runs one take at a time, cues every step with a beep and the step's words
   on the camera window (and in the console), records the glove at full rate
   plus the camera, and writes next to the frames an `events` file with the
   cue time, step or cycle-phase index, label and the expected flexed-finger
   set, plus every accept or reject decision with its time. During a take
   the window shows only the cue words and a countdown (no live skeleton,
   see the operator-bias note in section 4). One hand per run
   (`--hand left|right`), the other hand rests on the table. Sequence rounds
   are shuffled with a saved seed (D7).
3. `scripts/check_protocol.py`: pass/fail table per take (rules in section 4)
   so bad takes are redone in the same session, like `check_take_labels.py`
   does for the six poses.
4. Small additions to `record_poses.py` (camera): pose list from a file with
   the paper reference for each grasp, one hand-cropped still per take, and a
   per-take medoid keypoint file in the professor's format.
5. `scripts/package_professor_set.py`: builds the hand-in folder (section 5).

Tests with the existing mocks (`--mock`, `--mock-glove`, `--mock-leap`) for
2, 3 and 4. Estimated: one Opus agent, about half a day, then one real
session of each kind.

## 2. Decisions in the plan (each one is a question the review or the professor can overturn)

D1. Set A is mimed in the air, no object in the hand, as the primary
    protocol. The tracker loses fingers wrapped around an object, and the
    professor wants the hand configuration. But a mimed pose is not always
    the depicted grasp: in the Feix taxonomy several grasps are defined by
    the object's shape and size. So whether object takes are added is the
    professor's call (question 2 in section 7), not something gated on a
    detection percentage; if he wants them, they are a second pass with the
    real object and their own label suffix.
D2. Set A uses the bare LEFT hand, like the 21 camera poses handed in on
    2026-09-23 and the 81 glove frames from July. The operator's hand goes in
    the file name and meta; the tracker's own left/right label is kept in the
    data but not trusted (it called the left hand "right" in 20 of 21 poses).
D3. Set A orientation: not palm-to-lens for every grasp. The rule is an
    orientation envelope: hand 20 to 40 cm above the module, wrist inside the
    field of view, every finger chain visible to the lens, no finger edge-on.
    Grasps that are naturally palm-down (lateral/key, hook, extension) are
    rotated as little as needed to satisfy that, and the rotation used is
    written in the take's meta. The still per take shows what was recorded.
D4. Set A: 3 takes per grasp, 5 s each, 5 s preparation. Every take keeps all
    frames. The professor wants the final configuration, so each take gets an
    explicit static interval: the 2 s window with the least joint motion
    inside the take, written to the meta as start and end times. The
    one-frame summary is the medoid of that window (the real frame closest
    to the window's aligned mean), not of the whole take, so a hand that
    drifts into the pose during the first second cannot pull the summary.
D5. Sets B and C are recorded one hand at a time with the camera running as
    the reference, gloves on, palm to the lens, 18 to 28 cm above the
    module. Left hand first, then right. The professor only asked for glove
    data; the camera costs nothing extra, verifies each step, and gives lag
    at two speeds (the left glove trails by about 0.10 s, the right by about
    0.47 s). No mirrored two-hand fallback: it changes the geometry the
    camera sees and the two hands would be cued from one clock. If time is
    short, the fallback is the same protocol glove only, still one hand at a
    time, and the camera columns are simply absent from the report.
D6. Set B timing per cycle: bend 4 s, hold 1 s, straighten 4 s, rest 1 s;
    5 cycles in one take, so one take per finger (five takes), then index
    slow (bend 6 s, hold 1 s, straighten 6 s, rest 1 s, 5 cycles, so it is
    measurably slower than the normal take) and index fast (bend 1 s, hold
    0.5 s, straighten 1 s, rest 0.5 s, 8 cycles). Seven takes per hand. Every phase
    of every cycle (bend, hold, straighten, rest) is a cued event with its
    own timestamp in the events file, so a missed cue, fatigue or creep can
    be located to the cycle, and the checker counts cycles from the events
    as well as from the curl peaks. The 1 s rest is the reset between
    cycles; the protocol JSON states the repetition count explicitly so the
    checker verifies "5 repetitions" rather than assuming it.
D7. Set C timing: each configuration is cued by a beep and held 2.5 s (the
    first second is the movement, the rest is the hold; the hold window used
    for checking is the last 1.5 s). Sequence 7 (flex then release) has 11
    steps counting the opening open hand, the others 6 to 8. 3 takes per sequence, 21 takes per hand,
    recorded as three rounds: each round runs all seven sequences in a
    shuffled order (seed saved in the session meta), so the three takes of a
    sequence are independent repetitions and practice or fatigue cannot be
    mistaken for a glove effect. The seven sequences are spelled out step by
    step in `sequences.json`; the events file names the sequence, the step
    index and the expected flexed set for every cue.
D8. Both gloves are calibrated in XR Trainer at the start of the glove
    session, then a 6 s open/fist warm-up per hand records each finger's own
    open and fist endpoints for that session (they are not assumed equal
    across fingers or sessions). No recalibration of the values afterwards
    (the camera-referenced recalibration stays off, per the day 3 verdict).
    The session meta stores: session id, XR Trainer calibration time, the
    per-finger warm-up endpoints, the protocol file and its version, the
    operator's hand, and for every rejected take the rejection reason. Each
    take carries its take id and session id in every frame.
D9. One session per set is the deliverable. A second-day repeat of Sets B
    and C is planned if time permits, because the right glove's too-open
    fault is session-dependent and only a repeat separates repeatability
    from noise; it is not required for the hand-in.

## 3. Grasp list for Set A (from the three papers, 2026-09-28)

The professor supplied three papers; they are kept outside the repo in
`xr trainer\reference\grasp papers\` (paper1.pdf, paper2.pdf, paper3.pdf)
with a two-page `grasp sheet.pdf` that shows the figure panel to copy for
every id. `protocols/grasps.json` (version 2) holds 23 grasps under the
papers' own names:

- Paper 1, Heumer et al. 2007 (Schlesinger's taxonomy, Figure 1, Table 1):
  cylindrical, hook, lateral, palmar, spherical, tip. Six.
- Paper 2, de Souza et al. 2015 (Table 1, Fig. 13): scenarios 1 to 10:
  power grasp (cylinder), power grasp (ball), power grasp including finger
  side, fingertip grasp, fingertip grasp with side support, tripod grasp,
  directional power grasp with thumb and finger side, with thumb and finger
  tips, with thumb, finger side and finger tips, power grasp with dexterous
  ability. Ten.
- Paper 3, Cobos et al. 2009 (Figure 4, Cutkosky classification, Figure 5):
  circular power; prismatic power medium wrap and heavy wrap; circular
  precision with thumb and 4, 2 or 1 fingers; prismatic precision. Seven.

Grasps that are the same hand shape under two papers' names (cylindrical =
scenario 1 = medium wrap; spherical = scenario 2 = circular power; tip =
scenario 4 = thumb and 1 finger; tripod = thumb and 2 fingers; lateral ~
scenario 7; palmar ~ thumb and 4 fingers) are still recorded under each
label, and the `same_as` field links them so the analysis can merge them.
23 grasps x 3 takes = 69 takes, about 25 minutes of recording. Every paper
shows its grasps on an object; the plan records them mimed (D1) until the
professor asks for an object pass.

## 4. Quality rules (what makes a take acceptable)

Two kinds of rule, kept apart on purpose. Acquisition gates are fixed
numbers that decide on the spot whether a take is usable at all. Measured
values are continuous numbers written to the report; where a band is needed
to flag a take for a second look, the band comes from that session's
warm-up and from the set itself, not from a number chosen in advance. A
fixed threshold is never presented as the definition of a correct movement.

Set A (camera):
- acquisition gate: at least 90 % of the take's frames tracked and no
  re-acquisition inside the static interval; this is the minimum, not the
  acceptance;
- acceptance: the hand-cropped still and the summary frame match the
  paper's figure by eye, fingertip by fingertip (a take can be 95 % tracked
  with one fingertip wrong); the operator confirms or redoes on the spot,
  and the decision with its time is written by the recorder, not by hand;
- measured: the summary frame's `grab_strength` and `pinch_strength`
  (LeapC's own 0 to 1 values, already saved in every frame) and per-finger
  curls go in a table; outliers within a grasp's three takes are flagged
  for a second look.

Set B (glove, camera reference):
- acquisition gate: the cued finger's glove curl spans at least 60 % of its
  own open-to-fist range from the warm-up (a screening number; the actual
  span per finger and per take is reported);
- measured, not failed: the other four fingers' spans as a fraction of
  their own ranges (the ring drags the middle and pinky along, and the glove
  senses flexion only, so a hard cutoff would reject real coupled motion);
- cycles counted two ways, from the cued events and from the glove curl
  peaks; a mismatch is flagged;
- where the camera followed the finger (camera span at least 0.50, the
  `finger_sweep` rule), the transfer curve, hysteresis and lag are reported
  per finger and per speed, with the number of cycles and frames that met
  the span rule; lag is estimated per cycle and per take, never one number
  per hand, because the left and right gloves differ by a factor of four.

Set C (glove, camera reference):
- initial bands: for each step, the median glove fraction over the hold
  window above 0.6 for fingers that should be flexed; a finger that should
  be straight fails only when it reads flexed (0.6 or more); between 0.3
  and 0.6 it is reported as coupling (the ring pulls its neighbours along
  and the glove senses flexion only), not failed; the continuous fractions
  are kept, and after the first hand the bands are re-derived from that
  session's warm-up and from the open-hand and full-fist steps every
  sequence begins and ends with;
- a take passes when every step passes for the cued fingers;
- the same check on the camera where it is trusted (existing trust gates),
  so a glove fault (the right glove's too-open readings) is reported as the
  glove being wrong, not the operator;
- takes that fail are redone immediately; failed takes are kept under
  `rejected/` with the reason, like the coached recorder does.

Operator bias, and what the recorder does about it: one person cues,
performs, watches the camera and judges the take, and can unconsciously
shape the movement so the glove and camera agree, or slow down to cover the
right glove's lag. So during Set B and C takes the camera window shows only
the cue words and a countdown, not the live skeletons; the cue times and
every accept or reject decision are timestamped by the recorder; the
operator follows the beep, never the screen. The Ultraleap tracker also
estimates parts of the hand it cannot see from what it saw before, so a
smooth camera trace is not proof the finger was observed; the trust gates
and the per-take tracked fraction stay in the report for that reason.

Known glove limits that will show up and are reported as such, not hidden:
the thumb has flexion only (no opposition); spread is a template constant;
the right glove sometimes reads too open on a session-dependent basis; curl
creeps during holds; lag differs by hand.

## 5. Deliverable folder (no photographs)

`xr trainer\grasp and flexion set for professor <date>\`

- `README.txt`: setup, hand, dates, counts, the quality rules, known limits,
  and the coordinate convention of the 21-landmark text files (wrist at the
  origin in millimetres for the glove, as in the earlier hand-ins; camera
  frame in millimetres with the wrist line given for the camera). The JSONL
  stays the richer source; the text files are derived from it.
- `grasps\`: per grasp, `<grasp>_left_takeN.jsonl` (26 joints, metres,
  camera frame), `<grasp>_left_takeN_keypoints.txt` (medoid frame in the
  professor's 21-landmark format) and `<grasp>_left_all_frames.txt`;
  `grasps_summary.csv` (grasp, take, frames, tracked %, grab, pinch, curls).
- `finger_flexion\`: per hand and finger, the glove JSONL at full rate, the
  events file, the professor-format text export, and `flexion_report.txt`
  (cycles found, range, hysteresis, lag, per finger and speed).
- `sequences\`: per hand and sequence, the same three files per take plus
  `sequence_check.csv` (step by step pass/fail for glove and camera).
- Camera reference files for B and C go in beside the glove files with a
  `camera_` prefix, same names.

Hand-cropped stills stay in the repo's ignored `recordings/` for our own
checking and are not handed in.

## 6. Sessions, in order (each line names its setup)

1. No hardware: get the papers, fill `protocols/grasps.json`, build the tools
   (section 1), run the mock tests.
2. Camera only, bare left hand: Set A, 3 takes per grasp, check the summary
   table and the stills after every grasp, redo outliers. Recording time is
   about 20 minutes for 12 grasps; budget the same again for checking and
   retakes.
3. Gloves on + camera, left hand then right hand: calibrate in XR Trainer,
   warm-up, Set B (7 takes per hand, about 7 minutes of recording), Set C
   (21 takes per hand, about 10 minutes of recording), `check_protocol.py`
   after each set and hand, redo failed takes before taking the gloves off.
   Nominal recording is under 40 minutes for both hands; plan an afternoon,
   half of it for checking, retakes and packaging, and do the checking as
   you go rather than at the end.
4. No hardware: package the folder, write the README, post to Teams.
5. Gloves on + camera, a later day, if time permits: repeat Sets B and C
   (D9).

## 7. Questions to put to the professor with the first batch

1. Which papers and which figures define the grasp set (names and count)?
   The labels will be the papers' own terms.
2. Mimed grasps in the air, or holding the object? Several grasps in the
   taxonomies are defined by the object, so this changes what is recorded.
   (Plan: mimed first; object takes as a second pass if wanted.)
3. One hand (left, as before) or both?
4. The same 21-landmark text format as the earlier hand-ins, plus the raw
   JSONL? (Plan: both.)

## 8. Review outcome (ChatGPT, 2026-09-27)

Accepted and folded in above:
- D1: no-object stays primary, but object takes are the professor's
  decision, not a detection-gated extra (grasp definitions depend on the
  object in Feix).
- D3: an orientation envelope instead of palm-to-lens for every grasp.
- D4: the summary frame comes from an explicit static interval, and the
  deliverable names that interval ("final configuration only").
- D5: keep one hand at a time with the camera; the mirrored two-hand
  fallback is dropped.
- D6: cycle-level events and an explicit repetition count in the protocol
  file so the checker verifies "5 repetitions".
- D7: the three takes are independent repetitions in shuffled sequence
  order.
- D8: session, take, hand, calibration, protocol version and rejection
  reason in the metadata; per-finger endpoints per session.
- D9: second-day repeat "if time permits" rather than "optional".
- Grasp labels: the papers' own terms, no substitutions.
- QC: fixed numbers only as acquisition gates; scientific QC from
  continuous values and session-derived bands; 90 % tracked is not the sole
  acceptance; the 25 % other-finger rule is descriptive; camera-span
  coverage is counted; lag per cycle and take.
- Time: nominal recording is short; the afternoon goes to checking,
  retakes and packaging, done as you go.
- Risk added: operator bias (cue-only display during takes, recorder
  timestamps every cue and decision); the tracker's inference of unseen
  parts is why trust gates stay in the report.

Not adopted: nothing. The placeholder grasp list was confirmed as a
provisional superset of Schlesinger's six with common Cutkosky and Feix
variants.
