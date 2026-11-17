# Protocol: the thumb TMC calibration take (`tmc_neutral`)

Written 2026-10-23. Not yet run on the camera: the mock path and the tests
are the only runs so far. A recording-protocol addition to the thumb TMC
estimate of `docs/tmc_estimate.md`; it changes nothing in that estimate's
maths. Project rules that hold here: before a step is run it is posted in the
xr-training ChatGPT thread and its corrections applied; every step names its
hardware setup (**No hardware**, **Camera only, no hand over it**, or
**Camera only, bare left hand**); one action per step; durations come from
timestamps, never from the commanded seconds. A step whose title says "only
when" is run only in that case; otherwise go on to the next step.

Code: `scripts\leap\record_poses.py --calibration tmc_neutral` (the recorder
mode), `src\leap_hand\tmc_calibration_take.py` (the loader that turns the take
into a `tmc_estimate.Calibration`), `tests\test_tmc_calibration_take.py`.

## Why a calibration take

The thumb TMC estimate measures the thumb metacarpal against the tracker's
palm, relative to a rest frame `R_rest`. Today `R_rest` is the open hand the
participant happened to hold before each transition cue (`docs\tmc_estimate.md`,
"Calibration"). That zero is incidental: nobody defined the thumb's position
in it, so the estimate differs from any anatomical TMC angle by an unmeasured
change of reference (assumption A3; after the abduction-then-flexion
decomposition this is pose-dependent and mixes the two angles, not a fixed
amount per angle: `docs\tmc_estimate.md`, "What a change of reference does",
corrected 2026-10-24), and the across-transition agreement of that open hand
(median 4.4, p90 7.8, max 20.0 degrees on 2026-10-23) is its floor
uncertainty.

A calibration take replaces the incidental open hand with a posture that is
defined before it is held, held on purpose, held still, and held again, so
that the zero is a stated posture and its repeatability is measured. It does
not, by itself, validate the estimate: see "What this take gives and does
not give" at the end.

## The posture (`tmc_neutral_v1`)

### What the operator does

The hand flat and open with the palm toward the camera, the four fingers
straight and together, the thumb straight and resting against the side of
the index finger in the plane of the palm, nothing pressed. In the recorder's
words (`leap_hand.tmc_calibration_take.POSE_DEFINITION`, written into every
take's sidecar as `pose_definition`), so the file and this page cannot drift
apart.

Operationally, under the camera: the module flat on the table with the
lenses up; the left hand 20 to 30 cm above it, palm down toward the lenses
(the "OPEN HAND, palm to the camera" of the grasp protocol); the fingers
straight and touching each other along their length; the thumb straight,
its whole length against the side of the index finger, lying in the same
plane as the palm, neither lifted toward the lens nor pushed under the
palm; muscles relaxed, nothing squeezed (a pressed thumb deforms the hand
the way a squeezed fist does, `docs\protocol_recording.md`, step 5). The
right hand rests on the table away from the module.

### Where it comes from, and what a camera cannot build

The anatomically defined neutral the plan asks for
(`docs\angle_coverage.md`, section 5) is the ISB's. Wu et al. 2005 (Journal
of Biomechanics 38, 981-992) give the trapezium a coordinate system adapted
from Cooney et al. 1981 (section 4.3.3: Y from the mid-point of the central
ridge of the trapezial saddle to the junction of trapezium, scaphoid and
trapezoid; X dorsal to volar, perpendicular to that ridge, through the
tubercle of the trapezium; Z perpendicular to both), each metacarpal a system
with its origin midway between base and head and Y along the head-to-base
line (section 4.3.4), and a joint coordinate system with e1 the proximal
segment's Z (flexion and extension), e3 the distal segment's Y (axial
rotation) and e2 floating (abduction and adduction) (section 4.4.1). The
neutral posture of the first carpometacarpal joint is, in their words, the
position where the trapezium's system (Cooney's) and the first metacarpal's
system are identical (section 4.4.1). Cheze et al. 2009 (Computer Methods in
Biomechanics and Biomedical Engineering 12(3), 277-282, online 2008; cited by
Paper 3 and by the plan as Cheze et al. 2008) build both segment systems from
functional anatomy and put flexion-extension about a trapezium axis and
abduction-adduction about a first-metacarpal axis; on the same data the
fixed-axes sequence of Cooney et al., the ISB's mobile axes and theirs differ
by 9 degrees in flexion-extension, 2 in abduction-adduction and 13 in axial
rotation. Their abstract gives no neutral position, and the full text was not
reachable from this machine on 2026-10-23 (the HAL copy is behind a bot
wall); what is said about it here is the abstract and the ISB text only.

A camera-only setup has none of those landmarks. LeapC fits a rigid palm
(the PALM joint stands for the carpus, assumption A5) and a thumb metacarpal
bone whose base and head are the model's, not palpated; the trapezium is
not tracked at all. So neither the Cooney/ISB trapezium frame nor Cheze's
functional axes can be built from a recording, and the neutral cannot be
fixed by bone frames. What a camera-only take can fix is a **posture**: this
protocol takes the starting position of clinical thumb goniometry, the thumb
lying beside the index finger in the plane of the palm, where radial
abduction (in the plane of the palm) and palmar abduction (out of it) both
read zero. That posture is reproducible by an operator without a second
person and is the reference clinical measurements of the thumb start from,
which is why it is the stand-in. The offset between it and the ISB
bone-aligned neutral is unknown and stays unknown here; every number against
this zero is written under `*_est_neutral`, never under a Paper 3 name.

### What a goniometer reading would add

A hand goniometer laid on the posture gives, in the clinical convention, how
far the operator's resting thumb actually is from that zero: a radial
abduction reading (goniometer in the plane of the palm, one arm along the
index metacarpal, the other along the thumb metacarpal) and a palmar
abduction reading (goniometer perpendicular to the palm). With those two
numbers the posture's own error is a measured quantity, typically a few
degrees at the resolution of a plastic goniometer, instead of an assumption
of zero; the loader keeps the text in the calibration's provenance so that
anyone reading an estimate sees it. It would also allow, later, a one-point
scale check: a second posture held at a known goniometer angle of radial
abduction, compared with the estimate's `aa` from the same frames. That
second posture is not part of this protocol.

What a goniometer does not add: the ISB neutral (it reads the clinical
convention, not Cooney's bone frame), the tracker's own error on the thumb,
and the mapping's scale and cross-talk. Those need an independent sensor
(Paper 3 used a Cyberglove). The reading is optional in the steps below; a
session without one is still a valid calibration take, with its zero's error
unmeasured and said so.

## The hold: duration from timestamps

The recorder commands a hold of 5 s (`--duration 5`, the grasp protocol's
take length). It beeps, discards the frames that queued behind the beep,
opens the file, records every frame for the commanded seconds, closes the
file and beeps again. The file therefore runs from the REC beep to the stop
beep, and every frame in it is a held frame: the posture is made during the
countdown and must already be still at the REC beep.

The duration that counts is not the commanded one. The loader reads each
take's hold as the span of the frames' own camera clock (`timestamp_us`,
the first to the last frame of the hand in the file; `replay_core` names any
fallback clock) and writes it as `hold_s`; the sidecar's `hold_start` and
`hold_end` (wall clock, as commanded) are kept beside it as
`hold_commanded_s` for cross-checking, never used in its place. A take whose
`hold_s` is under 2 s, or that holds fewer than 20 frames, is skipped and
listed with the reason (`MIN_HOLD_S`, `MIN_FRAMES`); the 2 s is the length of
the grasp protocol's static interval, the 20 frames the open-hand
calibration's own minimum. Within a take the frames' spread about the take's
mean (`spread_median_deg`, `spread_max_deg`) says whether the posture was
held: on the open-hand windows that spread was 0.05 to 2.19 degrees at the
median per transition.

## Repeats

Three takes per session (`--takes 3`). Before each take the hand comes off
the posture and is made again during the 5 s countdown, so that the three
takes are three independent makings of the posture, not one hold cut in
three. The loader takes one chordal mean per take and the chordal mean of
those as `R_rest`, so every take weighs once however long its hold, and it
reports the across-take agreement (the angle of each take's mean to
`R_rest`: median, p90, max, and which take is worst). That agreement is the
zero's repeatability and its floor uncertainty, the role the
across-transition agreement plays for the open-hand zero. Three is the
smallest number that gives a worst take and a median; more takes
(`--takes 5`) cost 50 s and narrow nothing in the maths, only in the
statistics, so three is the default and five the choice when the first
session's worst take stands out.

The session is re-recorded when its `agreement_max_deg` is above 4.4
degrees, the open-hand calibration's median across-transition agreement
on 2026-10-23: a posture made on purpose three times should agree at least
as well as incidental open hands did between cues. Every folder is kept,
including the re-recorded one's predecessor.

## What the take's metadata records

Per take, `<take>.meta.json` beside the JSONL (written by the recorder,
read by the loader):

| Key | What it is |
|-----|------------|
| `kind` | `tmc_neutral_calibration_take`; the loader refuses any other sidecar |
| `calibration`, `pose` | `tmc_neutral` (the `pose` label on every frame line is the same) |
| `pose_definition` | `id` (`tmc_neutral_v1`), `text` (the operator's posture), `zero` (what it stands in for and why), `document` (this page) |
| `hand` | `left` or `right`: the hand that held the posture, which the loader selects from the file |
| `take`, `takes` | this take's number and the session's count |
| `duration_s`, `prep_s` | the commanded hold and countdown |
| `hold_start`, `hold_end` | the commanded hold on the wall clock: the file's opening after the REC beep, and its closing |
| `frames`, `hands`, `reacquired` | lines written, which hands the file holds, hand-id changes during the hold |
| `hz` | frames kept per second: `0`, every frame |
| `operator` | who recorded, from `--operator` |
| `goniometer` | the reading as written on the sheet, from `--goniometer`; `null` when none was taken |
| `mock` | `true` for a rehearsal on synthetic hands, which never sits beside a real session |
| `recorder_rules`, `tool_commit` | the recorder's rules date and the repository commit it ran from |
| `session`, `file`, `recorded` | the folder name, the final file name, the wall-clock start |

Per session, `calibration.json`: the `tmc_estimate.Calibration` the loader
built (`R_rest` in full precision and as xyz angles, one record per take with
`hold_s`, `hold_commanded_s`, the clock used, hand-id changes, gaps, spread
and angle to the session mean, the agreement, the skipped takes with their
reasons, and the provenance text that names the posture definition, the
hold rule, the agreement and the goniometer note).

## Steps

### Step 1. No hardware: open PowerShell in the repo folder

```
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
```

Every command below runs from this folder.

### Step 2. No hardware: rehearse the take on synthetic hands

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --mock --calibration tmc_neutral --hand left --takes 2 --duration 3 --prep 1
```

Two takes of 3 s with a 1 s countdown, about 10 seconds. The console ends
with `TMC NEUTRAL CALIBRATION OK: takes=2 ...` and the folder under
`recordings\protocol_mock\tmc_neutral\`. The mock holds its cartoon open
palm, not the posture, and says so; its numbers mean nothing about a thumb.
The rehearsal checks that the files are written and read back.

### Step 3. No hardware: look at the rehearsal folder

```
Get-ChildItem "recordings\protocol_mock\tmc_neutral\<folder from step 2>"
```

Two `tmc_neutral_..._take1_...` and `..._take2_...` pairs, each a `.jsonl`
and a `.meta.json`, and one `calibration.json`.

### Step 4. Camera only, no hand over it: check the camera

```
.venv\Scripts\python.exe scripts\leap\check_setup.py
```

No line may say FAIL. If a line says FAIL, do what that line says, then run
this step again.

### Step 5. Camera only, bare left hand: practise the posture in the live view

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --live
```

Hold the left hand 20 to 30 cm above the module, palm down toward it, and
make the posture of "What the operator does" for about ten seconds while
watching the window: the thumb's three bones drawn along the side of the
index, in the plane of the other fingers, and the thumb numbers steady.
Press q to close the window.

### Step 6. No hardware: only when a goniometer and a second person are at hand: read the posture on the table

The operator rests the left hand flat on the table, palm down, in the
posture. The second person reads radial abduction (goniometer flat on the
back of the hand, one arm along the index metacarpal, the other along the
thumb metacarpal) and palmar abduction (goniometer on its edge,
perpendicular to the palm, the same two arms) and writes one line on the
sheet: both readings in degrees, the instrument, their name.

### Step 7. Camera only, bare left hand: record the calibration take

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --calibration tmc_neutral --hand left --operator "N Kim"
```

It waits until it sees a hand, then runs three takes. Each take: a 5 s
countdown with three beeps at the end (make the posture now and be still
before the last beep), the REC beep (hold it, 5 s), the stop beep (lift the
hand off the posture). Between takes the hand comes off the posture and is
made again. About 30 seconds. The console ends with the summary, the line
`TMC NEUTRAL CALIBRATION OK: takes=3 frames=... hold_s=... agreement_max_deg=... skipped=0`,
the folder under `recordings\protocol\tmc_neutral\` and the path of its
`calibration.json`.

### Step 8. Camera only, bare left hand: only when step 6 was done: record a second session with the reading

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --calibration tmc_neutral --hand left --operator "N Kim" --goniometer "<the sheet's line from step 6>"
```

The same run as step 7, with the sheet's line written into every take's
sidecar and into the calibration's provenance. Both sessions are kept.

### Step 9. No hardware: compare the agreement with the open-hand floor

Read the `TMC NEUTRAL CALIBRATION OK` line of step 7 (and of step 8 when it
ran). `agreement_max_deg` is the worst take's angle to the session mean.
When it is above 4.4, the session is recorded again (step 10); when the
console said `TMC NEUTRAL CALIBRATION FAILED:` instead, the reason it gives
(too few frames, a hold under 2 s, no left-hand frames) is fixed and the
session is recorded again (step 10).

### Step 10. Camera only, bare left hand: only when step 9 said to record again: record the session again

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --calibration tmc_neutral --hand left --operator "N Kim"
```

The same as step 7. The earlier folder is kept; the hand-over names the
folder that is used.

## What the files are

```
recordings\protocol\tmc_neutral\<YYYYMMDD_HHMMSS>_<hand>\
  tmc_neutral_<hand>_take<N>_<stamp>.jsonl       every frame of the hold, the camera
                                                   line format, pose "tmc_neutral"
  tmc_neutral_<hand>_take<N>_<stamp>.meta.json   the sidecar of the table above
  calibration.json                               the tmc_estimate.Calibration
```

A rehearsal writes the same layout under `recordings\protocol_mock\`, its
sidecars saying `"mock": true`. The hand tag in a file name is the tracker's
(`both` when the mock or an idle hand put a second hand in the file); the
loader selects the hand the sidecar names. To use the calibration:

```
from leap_hand import tmc_calibration_take, tmc_estimate
cal = tmc_calibration_take.calibrate(r"recordings\protocol\tmc_neutral\<folder>")
angles = tmc_calibration_take.estimate(pos, rot, "left", cal)   # T_TMC_fe_est_neutral, ...
```

or `tmc_estimate.Calibration.from_json(json.load(open(".../calibration.json")))`
and `tmc_estimate.estimate` against it; `docs\tmc_estimate.md`, "A
calibration take instead of the open hand".

## What this take gives and does not give

Gives: a zero that is a stated, reproducible posture instead of an
incidental open hand; the posture's repeatability across takes as a measured
floor; the hold's length from the frames' own clock; and, with a goniometer,
the posture's own error in the clinical convention.

Does not give: the ISB bone-aligned neutral (no trapezium, no palpated
metacarpal landmarks), so the offset to Paper 3's zero stays unknown and
`T_TMC_fe` and `T_TMC_aa` stay `unavailable`; the mapping's scale and
cross-talk; the tracker's own error on the thumb. Those two need an
independent sensor, and no take recorded under this protocol claims them.
