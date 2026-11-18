# Protocol: a live validation capture of the thumb TMC and the finger CMC

Written 2026-10-24. Not yet run on hardware: no reference device, participant
or operator was available when it was written. Everything below that needs
no hardware has been run (steps 2 and 3); the rest is the procedure for when
the equipment is there.

What it produces: one tracker session (Ultraleap camera, bare hand) and one
reference capture of the same hand by an independent device, recorded at the
same time, then compared by `scripts/leap/reference_validation.py`
(`docs/reference_validation.md`). What each coordinate means and why each
movement is in the session: `docs/measurement_specification.md`.

## Required equipment

- The Ultraleap Stereo IR 170 on the desk, lenses up, its service running
  (the same setup as the grasp sessions, `docs/grasp_recording.md`).
- An independent reference device that measures the first metacarpal against
  the carpus (and, for the CMC, each metacarpal against the carpus). One of:
  - **optical motion capture** (six or more cameras), 3 to 4 mm markers on
    the dorsal first metacarpal (base and head), a rigid cluster on the
    dorsum over the second and third metacarpals (the carpal frame), markers
    on the fourth and fifth metacarpal heads, and the device's static
    calibration trial with palpated landmarks;
  - **an electromagnetic tracker** with a sensor on the dorsal first
    metacarpal and one on the dorsum over the second and third metacarpals
    (one more per metacarpal for the CMC);
  - **at minimum, a finger goniometer and a second person**, reading each
    static hold in the clinical convention (radial abduction in the plane of
    the palm, palmar abduction at a right angle to it). Holds only, no
    sweeps, no CMC.
- A converter from that device's export to the `hand_reference_v1` folder
  (`docs/reference_validation.md`, "Converting a device's export"). Not
  written yet: it depends on the device chosen.
- A participant and an operator (the second person reads the goniometer or
  runs the reference device).

## Setup and calibration

- The reference device is calibrated by its own procedure (camera wand,
  field mapping or the goniometer's zero), and its calibration has an id the
  reference capture will name (`calibration.id`, `calibration.provenance`).
- The ISB-style segment frames, or the device's own, are built from its
  static calibration trial; the rotation order and the positive directions
  of its two TMC angles are read from the device's documentation and written
  into the capture's header (they cannot be checked from the data:
  `docs/measurement_specification.md`, "Rotation order").
- Both devices use infrared. Motion-capture strobes or markers can disturb
  the Ultraleap's images, and metal near an electromagnetic field generator
  can bend its field: steps 6 and 7 check the tracker under the real setup
  before the session.
- The tracker calibration for the unvalidated estimate (optional): a
  `tmc_neutral` session of the same hand (`docs/tmc_calibration_protocol.md`),
  given to the comparison as `--tmc-calibration`.

## The session (`tmc_cmc_validation_v1`)

Start taps, then three rounds of every trial item (each round holds both
isolated sweeps, so that a fit by trial is possible), then end taps. 23 takes,
about 5 minutes with the 5 s countdowns. The reference device records
continuously from before the start taps to after the end taps.

| Item | Seconds | Isolated coordinate | What the participant does |
|---|---|---|---|
| `sync_start` | 8 | (sync) | tap the thumb tip on the index fingertip 3 times, about one second apart, fingers spread wide between taps; then the hand open and still |
| `tmc_neutral_hold` | 5 | none (hold) | hold the tmc_neutral posture: hand flat, palm to the camera, thumb straight against the side of the index in the plane of the palm |
| `tmc_flexion_sweep` | 10 | flexion_extension | thumb straight, in the plane of the palm: out to the radial side, then across the palm toward the base of the little finger, and back; twice, slowly; not lifted out of the plane |
| `tmc_abduction_sweep` | 10 | abduction_adduction | thumb straight, lifted away from the palm at a right angle to it and back to the index; twice, slowly; not moved sideways in the plane |
| `tmc_hold_extended` | 5 | none (hold) | thumb out to the side in the plane of the palm, about halfway, still |
| `tmc_hold_flexed` | 5 | none (hold) | thumb across the palm toward the little finger's base, about halfway, not touching the palm, still |
| `tmc_hold_abducted` | 5 | none (hold) | thumb lifted away from the palm at a right angle, about halfway, still |
| `cmc_palm_arc` | 8 | flexion_extension (CMC) | cup the palm, bringing the bases of the thumb and the little finger toward each other, then flatten it; twice, slowly |
| `sync_end` | 8 | (sync) | the same 3 taps, then the hand open and still |

The text the recorder prints and writes into every sidecar is
`leap_hand.reference_validation.CAPTURE_ITEMS`; this table repeats it.
Repeated holds and repeated trials: every hold and both sweeps are recorded
three times, the hand lifted and the posture remade between takes; the
comparison reports each trial's bias and the spread of those biases.

## Steps

### Step 1. No hardware: open PowerShell in the repository folder

```powershell
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
```

### Step 2. No hardware: rehearse the session on synthetic hands

```powershell
.venv\Scripts\python.exe scripts\leap\record_validation.py --participant REHEARSAL --hand left --reference-device "none (mock rehearsal)" --mock --trials 1 --duration 1 --prep 0
```

Expect `VALIDATION CAPTURE DONE: participant=REHEARSAL hand=left ... takes=9/9 mock=yes`.
It writes into `recordings\protocol_mock\validation\`, never among real sessions.

### Step 3. No hardware: run the end-to-end example

```powershell
.venv\Scripts\python.exe scripts\leap\reference_validation.py example --out results\reference_validation\example
```

Expect `EXAMPLE DONE: data_origin=synthetic status=synthetic_demonstration ...`.
Every file it writes says SYNTHETIC; it shows the files a real comparison
produces.

### Step 4. Reference device only: calibrate the reference device

Follow the device's own calibration procedure and write down its calibration
id and date. No command here: it is the device's software.

### Step 5. Reference device on the hand: place the markers or sensors and record the static trial

Place them as listed under "Required equipment" on the participant's hand
(the hand the session is for), then record the device's static calibration
trial with the landmarks its software asks for. No command here.

### Step 6. Camera and reference device, bare hand with markers or sensors: a one-round check session

With the reference device running (strobes on, field on), record one short
round:

```powershell
.venv\Scripts\python.exe scripts\leap\record_validation.py --participant P01 --hand left --reference-device "<make model serial, software version>" --operator "<name>" --trials 1 --duration 3
```

The recorder prints the take summary; a take with 0 frames, or many
re-acquisitions, means the tracker is disturbed by the setup: stop and fix it
(move the module, lower the strobe power, move metal away) before step 8.

### Step 7. No hardware: check that the tracker saw the sync taps in the check session

```powershell
.venv\Scripts\python.exe scripts\leap\reference_validation.py taps --session recordings\protocol\validation\<check session folder>
```

Expect `SYNC TAPS OK: 2 sync takes, 3 taps each`. `SYNC TAPS CHECK FAILED`
names the take with the wrong count: tap more clearly (tips touching, fingers
spread wide between taps) and repeat steps 6 and 7.

### Step 8. Camera and reference device, bare hand with markers or sensors: record the session

Start the reference device's recording first, then:

```powershell
.venv\Scripts\python.exe scripts\leap\record_validation.py --participant P01 --hand left --reference-device "<make model serial, software version>" --operator "<name>"
```

Stop the reference device's recording only after `VALIDATION CAPTURE DONE`.
`VALIDATION CAPTURE INCOMPLETE` lists the missing takes; a session with a
missing sync take or a whole missing round is recorded again.

### Step 9. No hardware: check the sync taps of the session

```powershell
.venv\Scripts\python.exe scripts\leap\reference_validation.py taps --session recordings\protocol\validation\<session folder>
```

Expect `SYNC TAPS OK`. Pair the six taps it lists with the six the reference
device saw (in order) in the reference header's `sync.events`, each as
`tracker_file` + `tap_index`.

### Step 10. No hardware: convert the reference device's export

Run the device's converter (outstanding: see "What remains outstanding") so
that it writes a `hand_reference_v1` folder: `reference.json`, `samples.csv`,
and the raw exports under `source\`, with `participant.id` P01, the session
folder's name as `session`, the hand, the device, its calibration, the
units, the frames, the rotation order, the signs and the sync events.

### Step 11. No hardware: check the reference folder

```powershell
.venv\Scripts\python.exe scripts\leap\reference_validation.py check recordings\validation\reference\<session folder>
```

Expect `REFERENCE OK: ... status=evaluated`. `REFERENCE REFUSED` says which
field to fix (missing units, an ambiguous sign, a missing rotation order,
another hand, a samples file edited after the header).

### Step 12. No hardware: compare

```powershell
.venv\Scripts\python.exe scripts\leap\reference_validation.py compare --reference recordings\validation\reference\<session folder> --tracker-session recordings\protocol\validation\<session folder> --calibration-trials 1 --out results\reference_validation\<session folder>
```

The mapping is fitted on trial 1 and evaluated on trials 2 and 3. Add
`--tmc-calibration <tmc_neutral session>\calibration.json` to evaluate the
unvalidated estimate beside it. Expect `COMPARISON DONE: data_origin=measured
status=evaluated ...` and read `report.md`.

### Step 13. No hardware: look at the mapped values in the viewer

```powershell
.venv\Scripts\python.exe scripts\leap\replay_view.py --file recordings\protocol\validation\<session folder>\tmc_flexion_sweep_left_take2_<YYYYMMDD_HHMMSS>.jsonl --hand left --tmc-mapping results\reference_validation\<session folder>\mapping.json
```

(the take's file name as `validation_session.json` lists it for
`tmc_flexion_sweep`, take 2)

The angle panel shows `mapped (reference-evaluated): T_TMC_fe_mapped ...`
inside the tested range, and the tested range and scope below it.

### Step 14. Camera and reference device, bare hand with markers or sensors: repeat on another day

Repeat steps 4 to 12 on another day (markers or sensors placed again) under a
new session, then compare the two `report.json` files: a mapping is only
evaluated for the session it was fitted in, and a mapping that is to be used
on other sessions needs its errors shown on sessions it was not fitted on.

## Synchronization checks

- Step 7 and step 9: exactly 3 taps detected in each sync take (thumb and
  index tips closer than 25 mm, apart again beyond 45 mm, within one tracking
  segment; `leap_hand.reference_validation.tap_times`).
- In the comparison: the six paired events give the clock offset and drift
  by least squares; any event further than the declared
  `sync.uncertainty_s` from that line, or a drift beyond 1 %, refuses the
  comparison (mispaired taps). With one event only the offset is known and
  the report says drift is not measured.
- The report reruns the held-out errors with the clock shifted by minus and
  plus the declared uncertainty: if they move a lot, the synchronization is
  the limit of the result, and the report says by how much.

## Quality checks and rejection reasons

| Check | Where | What happens |
|---|---|---|
| a take with no frames, or many re-acquisitions | recorder, step 6 or 8 | fix the setup, record again |
| sync taps miscounted | step 7, step 9 | tap again, record again |
| events disagree beyond the declared uncertainty, or drift beyond 1 % | comparison | refused: fix the pairing |
| reference sample marked invalid (occluded marker, sensor dropout) | the device's export, `valid` 0 with its reason | excluded, `reference_invalid: <reason>` |
| a reference coordinate missing | samples | excluded, `reference_coordinate_missing` |
| no tracker take covers the sample's time | comparison | excluded, `outside_tracker_takes` |
| the sample falls in a take of another item | comparison | excluded, `movement_mismatch` (a sync or labelling error) |
| the tracker frames around the sample are cut by a gap, an identity change or an invalid-status line | comparison | excluded, `tracker_discontinuity: <which>`; never interpolated across |
| a hold that crosses such a cut, or has fewer than 10 tracker frames | comparison | excluded, `tracker_discontinuity inside the hold` / `hold_too_short` |
| a take off the camera clock | comparison | excluded, `tracker_clock_fallback` |
| the calibration trial lacks an isolated flexion or abduction sweep | comparison | mapping refused: record both in every trial |
| a sweep that is not isolated | report: the reference's own off-axis slope | read it before reading the cross-talk; record again when it is as large as the cross-talk being measured (the limit for that awaits agreement) |

Every exclusion is a row of `matched_samples.csv` with its reason, and is
counted in the report.

## Expected output files

| Folder | Files |
|---|---|
| `recordings\protocol\validation\<YYYYMMDD_HHMMSS>_<hand>\` | one `<item>_<hands>_take<n>_<YYYYMMDD_HHMMSS>.jsonl` per take (`<hands>`: the hands the camera saw, `left`, `right` or `both`; the recorder's naming, `xr_hand.recorder`), each with its `.meta.json` sidecar (participant, hand, session, item, instruction, reference device, commanded start and end); `validation_session.json` (every take with its sha256) |
| `recordings\validation\reference\<session>\` | `reference.json`, `samples.csv`, `source\` (the device's raw exports, hashed in the header) |
| `results\reference_validation\<session>\` | `report.json`, `report.md`, `matched_samples.csv`, `mapping.json` |

The recordings are never edited after the session: the comparison refuses a
take whose sha256 no longer matches the manifest, and a samples file whose
sha256 no longer matches the header.

## What remains outstanding

- **The reference device**: chosen, available and calibrated. None is in
  this repository or on this desk on 2026-10-24.
- **The converter** from that device's export to `hand_reference_v1`.
- **A participant session**: steps 4 to 13 with an operator.
- **The interference check** of step 6 with the real device.
- **An agreed acceptance threshold**, if a mapping is to be accepted or
  rejected rather than reported: none is sourced, and Paper 3 gives none.
- **Cross-session repeatability** (step 14) before a mapping is used on any
  session other than its own.
- **The finger CMC** need markers or sensors on each metacarpal; even then
  the values are the reference device's, because this tracker does not move
  its metacarpals.
