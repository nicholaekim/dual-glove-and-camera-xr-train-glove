# Reference measurements: the file format and the comparison

Written 2026-10-24. Code: `src/leap_hand/reference_validation.py` (module
docstring), command line `scripts/leap/reference_validation.py`, capture
`scripts/leap/record_validation.py`, tests `tests/test_reference_validation.py`
and `tests/test_record_validation.py`. What each coordinate is:
`docs/measurement_specification.md`. The live procedure:
`docs/validation_capture_protocol.md`.

**No reference measurement exists yet.** The format, the importer and the
comparison are ready; every result on this page comes from the synthetic
example, which measures nothing and says so in every file it writes.

## What goes in, what comes out

```
tracker session (record_validation.py)        reference capture (the device's converter)
  <item>_<hands>_take<n>_<stamp>.jsonl + .meta.json   reference.json   header: who, device,
  validation_session.json  (sha256 of each)                      units, frames, order, signs,
                                                                 sync, calibration
                                                samples.csv      one row per sample or hold
                                                source/          the device's raw exports
                    \                              /
                     reference_validation.py compare --calibration-trials 1
                                        |
             report.json  report.md  matched_samples.csv  mapping.json
```

The reference capture lives beside the tracker recordings, never inside
them; nothing is written into a recording or the dataset.

## The reference capture (`hand_reference_v1`)

A folder with `reference.json`, the samples CSV it names and the source
files it lists. `scripts/leap/reference_validation.py check <folder>` reads
and checks it; every refusal names the field and the fix (all of them are
tests in `tests/test_reference_validation.py`, group `format`).

### `reference.json`

| Field | Required | Meaning; refused when |
|---|---|---|
| `format` | yes | `"hand_reference_v1"` |
| `data_origin` | yes | `measured`, `synthetic` or `software_cross_check` |
| `role` | yes | `anatomical_reference` (segment frames from landmarks: optical or electromagnetic, or clinical goniometry), `independent_sensor` (an independent device that reports its own model's angles, e.g. a glove) or `software_cross_check`. An instrumented glove as `anatomical_reference` is refused |
| `participant.id` | yes | the participant, as the tracker session names them |
| `hand` | yes | `left` or `right` |
| `session` | yes | the tracker session folder's name |
| `device` | yes | `kind` (`optical_motion_capture`, `electromagnetic_tracker`, `goniometer`, `inertial_sensors`, `instrumented_glove`, `other`), `make`, `model`, `serial`, `software`, `sample_rate_hz`, and `derived_from_tracker` (true or false, required). True is accepted only with `data_origin` and `role` both `software_cross_check`: a second calculation from the tracker's own skeleton is a software cross-check, never a reference |
| `calibration` | yes | the reference device's own calibration: `id`, `provenance`, optional `hand` (refused when it is the other hand) |
| `tracker_calibration` | no | the tracker calibration the unvalidated estimate is compared with: `file`, `sha256`, `kind`, `hand`. When present, `compare --tmc-calibration` must give that very file (sha256), of the capture's hand |
| `units.angle` | yes | `deg` or `rad`; refused when missing or anything else |
| `clock` | yes unless `per_take` | `name`, `units` (`s`, `ms`, `us`): the unit of `time_s`, `time_end_s` and the sync times |
| `sync` | yes | `method` (below), `uncertainty_s` (a number >= 0, required unless `per_take`), `events` |
| `frames.<joint>` | per joint used | `proximal_segment`, `distal_segment`, `proximal_frame`, `distal_frame` (text), `zero` (`id`: `isb_neutral`, `clinical_zero`, `device_static_trial` or `other`, and `text`), `rotation_order` (thumb: `flexion_then_abduction` or `abduction_then_flexion`, refused when missing), `positive` (per movement: thumb flexion `ulnar` or `radial`, thumb abduction `palmar` or `dorsal`, CMC flexion `palmar` or `dorsal`; any other word is refused as ambiguous) |
| `coordinates` | yes | one per column: `column`, `joint` (`thumb_tmc`, `index_cmc`, `middle_cmc`, `ring_cmc`, `little_cmc`), `movement` (`flexion_extension`, `abduction_adduction` for the thumb). A thumb needs both of its columns |
| `movements` | yes | one per movement id the samples name: `id`, `joints` (the joints it is compared on), `isolated` (the coordinate an isolated movement moves, or null for holds). The capture protocol's item ids |
| `samples` | yes | `file` and `sha256`; refused when the file changed after the header was written |
| `source_files` | yes | the device's raw exports: `path` (inside the folder), `sha256`, `description` |

### `samples.csv`

| Column | Meaning |
|---|---|
| `sample_id` | unique, not empty |
| `trial` | 1, 2, 3 ...: the round of the capture (the split is by trial) |
| `movement` | a movement id of the header; empty = unlabelled (excluded) |
| `time_s` | the sample's time on the reference clock (`clock.units`); required unless `per_take` |
| `time_end_s` | empty for an instant; for a hold, its end: the sample is then compared with the tracker's mean over the interval |
| `tracker_file` | `per_take` only: the tracker take the hold was read on (the whole take is the hold) |
| `valid` | 1 or 0; 0 needs `exclusion_reason` (occluded marker, sensor dropout ...) |
| `exclusion_reason` | why the device marked it invalid |
| one column per coordinate | the value in `units.angle`; empty = not measured in this sample |

### The canonical convention

Every reference value is converted on import to one convention (module
docstring, "CANONICAL CONVENTION"): in the proximal segment's frame at the
declared zero, e3 along the metacarpal toward its head, e2 the palmar normal,
e1 = e2 x e3. Thumb flexion positive when the head moves ulnarly across the
palm, about e2; thumb abduction positive palmar, about e1; finger CMC flexion
positive palmar. A declared opposite sign is flipped, radians are converted,
and the declared rotation order is used to turn the two thumb angles into the
metacarpal's direction and back. All errors are reported in this convention.

### Synchronization (`sync.method`)

| Method | When | How the reference clock reaches the tracker's (`timestamp_us`) |
|---|---|---|
| `shared_events` | sync taps seen by both devices | each event pairs a `reference_time_s` with `tracker_timestamp_us` or with `tracker_file` + `tap_index` (resolved from the taps the tracker detected in that take). One event: offset only. Two or more: offset and drift by least squares, drift within 1 %, else refused. Three or more: every event within `uncertainty_s` of the line, else refused; one or two events leave no residual, so the report gives none and says the pairing is not checked |
| `hardware_trigger` | a shared trigger line | as `shared_events` (the trigger instants as events) |
| `same_clock` | the reference wrote the tracker's own clock | reference seconds are `timestamp_us` / 1e6 (in `clock.units`) |
| `per_take` | goniometer holds | no clock: each sample names its take; the whole take is the hold |

The declared uncertainty is reported and used: the comparison is run again
with the clock shifted by minus and plus it, and the held-out errors of the
three runs are in the report ("sync sensitivity").

## The comparison

`scripts/leap/reference_validation.py compare --reference <folder>
--tracker-session <folder> --calibration-trials 1 --out <folder>
[--tmc-calibration <calibration.json>]`. In order:

1. **One capture.** The reference's participant, hand and session must equal
   the tracker session's (refused, naming which differs). A tracker
   calibration must be of the same hand, with a known hand
   (`tmc_estimate.mismatch`), and the file the header names.
2. **The split.** `--calibration-trials` lists the trials the mapping is
   fitted on; every other trial is held out. No calibration trial, or no
   held-out trial, is refused. The fit sees only calibration samples; the
   metrics see only held-out samples; `report.json` lists both sets of
   sample ids, which never overlap.
3. **Matching.** Each (sample, joint the movement names) is placed on the
   tracker clock and matched to the tracker pose of the take that covers it:
   an instant to the pose interpolated between the two bracketing frames of
   one segment (`replay_core.pose_at`, never across a gap, an identity change
   or an invalid-status line); a hold to the chordal mean of its frames, which
   must lie in one segment. The tracker quantity is the bone against the
   palm, `R_pm = R_palm^T R_bone`.
4. **The mapping** (thumb TMC): `d = A R_pm b`, with `A` the rotation from the
   tracker's palm frame to the reference's proximal frame and `b` the
   metacarpal's anatomical long axis in the tracker's metacarpal frame,
   fitted on the calibration samples by Levenberg-Marquardt (five
   parameters) from the SVD solution of Wahba's problem. The fit reports its
   residual and each parameter's standard deviation. It is refused when the
   calibration samples do not span two movement directions on either side
   (one posture or one plane: a posture calibration fixes a zero, not a
   frame). `b` is a fitted parameter: with a gain or cross-talk in the
   tracker it moves away from the anatomical axis (19 deg in the synthetic
   example against 5 deg true), so it is judged by the held-out errors only.
5. **The metrics**, per movement and pooled, for each coordinate, on the
   held-out samples: n, signed bias, MAE, RMSE, absolute error p50 / p90 /
   p95 / max, signed p5 / p95, the reference's range (the tested range); for
   an isolated movement, the isolated coordinate's gain error (slope of its
   error against its reference value) and the cross-talk into the other
   (slope of the other's error against the isolated reference value, its
   range, and the reference's own off-axis slope); the repeatability (each
   trial's bias, their SD and range over two or more trials). Angle
   differences are wrapped into [-180, 180). With `--tmc-calibration`, the
   unvalidated estimate is evaluated on the same held-out samples, its
   direction read in the reference's order (`estimate_<kind>`).
6. **The finger CMC.** The tracker's metacarpal against the palm is measured
   over the samples; when it moves less than 0.1 deg the verdict is
   `unobservable`, nothing is fitted, and the report gives the reference's
   range beside it. The coordinate stays empty in every output.

### Exclusion reasons (`matched_samples.csv`, `report.json` `excluded_by_reason`)

| Reason | Meaning |
|---|---|
| `reference_invalid: <device's reason>` | the device marked the sample invalid |
| `reference_unlabeled` | no movement id |
| `reference_coordinate_missing: <column>` | a coordinate the joint needs is empty |
| `outside_tracker_takes` | no tracker take covers the time (between takes, before, after) |
| `movement_mismatch` | the sample's movement is not the item of the take it falls in |
| `tracker_clock_fallback` | that take is not on `timestamp_us`: no time can be placed on it |
| `tracker_discontinuity: <gap / id_change / invalid_status ...>` | the sample falls between two frames separated by a cut |
| `tracker_discontinuity inside the hold` | a hold crosses a cut |
| `hold_too_short` | fewer than 10 tracker frames in a hold |

### Thresholds

No accuracy threshold is applied. None is sourced or agreed for these
coordinates; Paper 3's couplings are model assumptions and its fingertip
errors are reconstruction position errors. A threshold for accepting a
mapping awaits agreement. The conditions the software does use are chosen
here and are not accuracy limits (`IDENTIFIABILITY_NOTE`): 5 deg RMS spread
on the second principal axis for a fit, 0.1 deg for "does not move", 10
frames for a hold, 1 % for a clock drift, 25 / 45 mm for a sync tap.

## Outputs and where a mapped value may appear

| File | Content |
|---|---|
| `report.json` | everything above, with the status: `evaluated` (measured, anatomical reference), `synthetic_demonstration` or `cross_check` |
| `report.md` | the same, readable |
| `matched_samples.csv` | one row per (sample, joint): role (calibration / heldout), used, reason, take, tracker times, frame ids, reference and mapped values, errors |
| `mapping.json` | `A`, `b`, the rotation order, the zero, the participant, hand, session, devices, the tested range (the held-out samples' reference range; the calibration range is in the report, fitted rather than tested), the held-out summary, the ids of the samples the fit and the metrics used, the provenance hashes, the status |

A mapping's values are **reference-evaluated only inside its scope**: the
participant, hand and tracker session it was evaluated on, inside its tested
range. `scripts/leap/replay_view.py --tmc-mapping mapping.json` shows them in
the angle panel's `mapped (...)` rows; `scripts/leap/replay_verify.py
--tmc-mapping mapping.json` writes them to `T_TMC_fe_mapped`,
`T_TMC_aa_mapped` with `T_TMC_mapped_status` (`evaluated_in_range`,
`synthetic_demonstration`, `cross_check`, or `outside_tested_range` with the
values empty) and `T_TMC_mapping`. Outside the scope every cell is empty and
the reason is given (`reference_validation.scope_mismatch`). The 2026-10-08
dataset has no participant column and another session than any validation
capture, so no mapping applies to it: one successful session does not turn
the historical estimates into measurements.

## The end-to-end example (synthetic)

```powershell
.venv\Scripts\python.exe scripts\leap\reference_validation.py example --out results\replay_fidelity\reference_validation_20261024\example
```

It writes a synthetic tracker session (23 takes of the protocol, 90 Hz, a
tracking gap in held-out flexion trial 2 and an invalid-status line in
held-out abduction trial 3), a `tmc_neutral` tracker calibration from its
neutral holds, and a synthetic reference capture (50 Hz, flexion declared
positive radial so the import has to flip it, a clock offset of 4995.25 s and
a drift of 40 ppm, six sync taps, ten occluded samples, one missing
coordinate, one mislabelled sample, samples between takes), then compares
them with trial 1 as the calibration trial. The tracker's thumb encodes the
truth with a gain of 0.93 on flexion, cross-talk 0.06 and 0.08 and 0.4 deg
noise; the reference adds 0.3 deg noise. Its output (2026-10-24):

```
EXAMPLE DONE: data_origin=synthetic status=synthetic_demonstration calibration_samples=774 heldout_samples=1520 excluded=41 mapping=tmcmap-477806454c8f mapped_fe_mae=0.93 mapped_aa_mae=0.60 estimate_neutral_fe_mae=4.16 estimate_neutral_aa_mae=6.26 ring_cmc=unobservable little_cmc=unobservable
```

The clock came back as offset 4995.250021 s and drift 4.38e-5 (residual max
1.4 ms against the declared 15 ms); the 41 exclusions are 17
`tracker_discontinuity`, 12 `outside_tracker_takes`, 10 `reference_invalid`,
1 `reference_coordinate_missing`, 1 `movement_mismatch`; the held-out flexion
sweep shows a gain error of -0.061 deg/deg (the 0.93 gain the rotation could
not absorb) and the neutral-posture estimate, which is a zero and not a frame,
is off by 4.2 / 6.3 deg. These are properties of the generator, not of a hand.

## Converting a device's export

Not written yet; it depends on the device. A converter must: read the
device's export without editing it, copy it under `source/` and hash it;
write one row per sample with the device's own validity flag and reason, in
the device's units with `units.angle` saying which; label each sample with
the protocol item and trial it belongs to (from the device's own event
marks, or by placing the tracker takes' times on the reference clock with
the sync taps); declare the device's frames, zero, rotation order and signs
from its documentation; list the six tap times as sync events; and write the
header last, with the sha256 of `samples.csv`. For a goniometer, the
converter is a typed sheet: one row per hold take, `sync.method` `per_take`,
`tracker_file` the take, `role` `anatomical_reference`, `device.kind`
`goniometer`, the clinical convention declared in `frames.thumb_tmc`.

## Limitations

- Nothing here has met real reference data. The importer's rules, the
  matching and the fit are exercised on synthetic captures with known
  answers only.
- The mapping is a rotation and an axis: a gain or a cross-talk of the
  tracker is reported, not corrected. A correction would need evidence that
  it is stable across sessions first.
- The rotation order cannot be checked from the data (a wrong declaration
  moves the errors by 0.12 deg in the test); it comes from the device's
  documentation.
- The sync taps are found by fixed distance thresholds on the tracker's own
  fingertips; a first real session must confirm they are counted right
  (protocol step 7).
- The finger CMC can only ever be the reference device's measurement: this
  tracker does not move its metacarpals.
