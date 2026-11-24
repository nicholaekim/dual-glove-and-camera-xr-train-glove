# Handoff for Codex: the validation and measurement workflow for the six unavailable coordinates

Written 2026-10-24. Repository `C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer`,
HEAD c777d4c plus uncommitted changes (nothing committed; the working tree
also holds the earlier corrections' uncommitted files and other sessions'
files). Ledger `GATES_reference_validation.md`; evidence
`results\replay_fidelity\reference_validation_20261024\` (README inside).
Codex's independent verification of this work has not been done.

**Gate status** (`node ~/.claude/skills/unlazy/scripts/gate-check.mjs --reverify
--timeout 3600 GATES_reference_validation.md`, after the last code change):
19 met (V1 to V18 re-run and passed, V19 by eye), 3 abandoned with their
reasons (H1 physical accuracy, H2 live capture, H3 finger CMC): the checker
ends `HANDOFF REQUIRED` and exits 1, as it must while those three are
outstanding. Log: `gate_reverify.log`.

**What this is and is not.** The six Paper 3 coordinates (`T_TMC_fe`,
`T_TMC_aa`, `I_CMC`, `M_CMC`, `R_CMC`, `L_CMC`) are still not measured and not
anatomically validated. This work adds the specification of what each needs,
the software path real reference measurements take (format, importer,
synchronization, matching, a mapping fitted on calibration trials and tested
on held-out trials, metrics, measurement status in the viewer, exports and
coverage table), a live capture procedure with a recorder, and one genuine
measurement on the existing data: the finger metacarpals never move against
the tracker's palm, so the finger CMC are unobservable by this tracker. No
reference device, participant session or reference recording exists; every
comparison result here is from synthetic data and labelled so.

## 1. Changed files and rationale

| File | State | Why |
|---|---|---|
| `src/leap_hand/reference_validation.py` | new | the format `hand_reference_v1` and its importer (refusals with the fix), the canonical TMC and CMC conventions, the tracker session reader (sha256 per take), clock alignment from paired events and detected thumb-index taps, sample matching that never crosses a discontinuity, the Levenberg-Marquardt fit of `A` (palm to proximal frame) and `b` (metacarpal axis) with identifiability refusals, metrics (bias, MAE, RMSE, percentiles, gain error, cross-talk, repeatability, tested range), the sync sensitivity rerun, CMC observability, the mapping's scope and application, the report writers, the capture protocol items, the synthetic example |
| `scripts/leap/reference_validation.py` | new | command line: `check`, `taps`, `compare`, `observability`, `example` |
| `scripts/leap/record_validation.py` | new | the tracker side of a validation session, on `record_poses.py`'s own `Session` (unchanged), with sidecars and a hashed manifest |
| `src/leap_hand/angle_coverage.py` | changed | status `evaluated` (a measurement evaluated against an independent reference, with its scope), labels `unvalidated estimate` / `reference-evaluated`, `STATUS_MEANINGS`, two entries for `T_TMC_fe_mapped` / `T_TMC_aa_mapped` (47 entries) |
| `src/leap_hand/replay_viewer.py` | changed | optional `mapping_provider` and a two-row `mapped (...)` block under the unavailable line; the title says "calculated = tracker coordinate"; the estimate rows say "unvalidated estimate"; the unavailable line fits one row; the three panels' bottom from 0.24 to 0.225 (one more row, clear of the timeline legend), so the worst case (interpolated pose, estimate, mapping) fits the 804 px window |
| `scripts/leap/replay_view.py` | changed | `--tmc-mapping`, `mapping_provider` (scope check, console lines) |
| `scripts/leap/replay_verify.py` | changed | `--tmc-mapping`; per-frame `T_TMC_fe_mapped`, `T_TMC_aa_mapped`, `T_TMC_mapped_status`, `T_TMC_mapping`; `tmc_mapping_used.json`; a summary section; the checker's `mapped_columns=2 mapped_filled=<m>` with negative controls |
| `tests/test_reference_validation.py` | new | 56 tests, groups spec, format, sync, compat, align, metrics, gaps, split |
| `tests/test_record_validation.py` | new | the recorder on the mock, the sequence, the protocol document's commands |
| `tests/test_angle_coverage.py`, `tests/test_replay_viewer.py`, `tests/test_replay_verify.py` | changed | 47 entries, the new labels and texts, the checker's longer line; new tests for the mapped block (explicit values from a hand-made mapping, scope refusals, tested range, real-window fit) and the mapped columns (scope, range, another session, checker negative controls) |
| `docs/measurement_specification.md` | new | Phase 1: the six, each with segments and axes, reference posture, rotation order, signs and units, what the tracker supplies, the additional measurement, the accuracy check; Paper 3's assumptions apart from what the dataset supports; the observability measurement; the device integrations inspected |
| `docs/reference_validation.md` | new | the format, the comparison, the exclusion reasons, the outputs, the scope of a mapping, the example, converters, limitations |
| `docs/validation_capture_protocol.md` | new | the live procedure: equipment, setup, the session's items, 14 steps each naming its setup, sync checks, quality checks and rejection reasons, expected files, what remains outstanding |
| `docs/angle_coverage.md`, `docs/tmc_estimate.md`, `README.md` | changed | the fifth status and the four kinds of value; the regenerated table; pointers to the new path |
| `GATES_reference_validation.md` | new | V1 to V19, and H1 to H3 abandoned with their reasons |

## 2. Exact commands and test results

From the repository root (the `CHECK:` lines of the ledger):

```
.venv\Scripts\python.exe -B -m pytest tests/test_replay_core.py tests/test_paper3.py tests/test_replay_viewer.py tests/test_replay_verify.py tests/test_angle_coverage.py tests/test_tmc_estimate.py tests/test_tmc_calibration_take.py tests/test_reference_validation.py tests/test_record_validation.py -q -p no:cacheprovider
```

Result (gate V16, and `focused_suite.log` in the evidence folder): `324 passed in 136.76s`. The new files alone: 56 tests in `tests/test_reference_validation.py` and 3 in `tests/test_record_validation.py`.

The groups (gates V1 to V9) are `-k spec`, `format`, `sync`, `compat`,
`align`, `metrics`, `gaps`, `split` of `tests/test_reference_validation.py`
and `-k "mapping or status or coverage"` over the coverage, viewer and batch
tests.

## 3. Full-suite results

```
.venv\Scripts\python.exe -m pytest tests -q -x -p no:cacheprovider
```

Result (gate V17, and `full_suite.log`): `1516 passed, 5 skipped in 1292.20s (0:21:32)`. No failure, so no regression to tell apart from an unrelated one. The earlier full runs of 2026-10-23 and 2026-10-24 also skipped 5; this run was quiet (`-q`), so it did not print their reasons.

## 4. Existing-dataset fidelity results

```
.venv\Scripts\python.exe scripts\leap\replay_verify.py --dataset "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\xr trainer\grasp set 2026-10-08 organized" --out results\replay_fidelity\reference_validation_20261024\batch --all --scenarios --check-angle-exports --tmc-mapping results\replay_fidelity\reference_validation_20261024\example\comparison\mapping.json
.venv\Scripts\python.exe scripts\leap\replay_verify.py --check-scenarios results\replay_fidelity\reference_validation_20261024\batch
.venv\Scripts\python.exe scripts\leap\replay_verify.py --check-dataset-unchanged "C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\baseline_sha256.json" "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\xr trainer\grasp set 2026-10-08 organized"
```

Results (gates V13 to V15):

```
FIDELITY OK: static=115 transitions=52 frames=71709 within_tolerance=167/167
ANGLE EXPORTS OK: rows=71709 mcp_columns=16 est_columns=4 unavailable_empty=6 est_filled=71709 mapped_columns=2 mapped_filled=0
SCENARIOS OK
DATASET UNCHANGED
```

The mapping given is the synthetic example's: its scope is another
participant's session, so it fills no dataset row (`mapped_filled=0`) and
`tmc_mapping_used.json` gives the reason for each of the 167 recordings.

## 5. The reference-data format and an end-to-end example

Format: `docs/reference_validation.md` ("The reference capture"). Example:

```
.venv\Scripts\python.exe scripts\leap\reference_validation.py example --out results\replay_fidelity\reference_validation_20261024\example
```

```
EXAMPLE DONE: data_origin=synthetic status=synthetic_demonstration calibration_samples=774 heldout_samples=1520 excluded=41 mapping=tmcmap-477806454c8f mapped_fe_mae=0.93 mapped_aa_mae=0.60 estimate_neutral_fe_mae=4.16 estimate_neutral_aa_mae=6.26 ring_cmc=unobservable little_cmc=unobservable
```

The example's tracker session, neutral calibration, reference folder and
comparison are all in that folder; every file says synthetic and the
mapping's status is `synthetic_demonstration`. It recovers the injected clock
(offset 4995.25 s, drift 40 ppm: 4995.250021 s, 4.38e-5), excludes every
planted defect with its reason (17 tracker discontinuities, 12 outside takes,
10 invalid reference samples, 1 missing coordinate, 1 mislabelled sample),
reports the gain the rotation cannot absorb (-0.061 deg/deg on the held-out
flexion sweep), and shows the neutral-posture estimate, a zero rather than a
frame, off by 4.2 / 6.3 deg.

## 6. The capture protocol

`docs/validation_capture_protocol.md`. Steps 2 and 3 (mock rehearsal, the
example) run here; steps 4 to 14 need the reference device, a participant and
an operator.

## 7. Joint-by-joint status

| Coordinate | The tracker gives | Status today | What would change it |
|---|---|---|---|
| `T_TMC_fe` | the thumb metacarpal against the rigid palm, moving (up to 40.25 deg within a recording) | `unavailable` under the Paper 3 name; `estimated` (unvalidated) under `T_TMC_fe_est_openhand` / `_neutral`; `T_TMC_fe_mapped` empty everywhere (no reference capture) | a reference capture and comparison: then `evaluated`, for that participant, hand, session and tested range only |
| `T_TMC_aa` | the same | the same | the same |
| `I_CMC` | nothing that moves (0.000023 deg at most) | `unavailable`, unobservable by this tracker | a device measuring the metacarpal against the carpus; the values would be that device's, never the tracker's |
| `M_CMC` | nothing that moves (0.000021 deg) | `unavailable`, unobservable | as above |
| `R_CMC` | nothing that moves (0.000021 deg) | `unavailable`, unobservable | as above |
| `L_CMC` | nothing that moves (0.000024 deg) | `unavailable`, unobservable | as above |

## 8. Measured validation results

None. No genuine reference data exists. The one genuine measurement made
here is the observability sweep of the dataset (gate V11):

```
OBSERVABILITY: recordings=167 frames=71709 thumb_max=40.248235deg index_max=0.000023deg middle_max=0.000021deg ring_max=0.000021deg little_max=0.000024deg thumb_median=4.63deg
VERDICT: finger_cmc=unobservable thumb_tmc=observable
```

## 9. Outstanding equipment, recordings and human actions

- A reference device (optical motion capture, an electromagnetic tracker, or
  at least a goniometer and a second person), calibrated (gate H1, abandoned
  here with that reason).
- A converter from its export to `hand_reference_v1`.
- A participant, an operator, and the live session of
  `docs/validation_capture_protocol.md` with the reference device recording
  at the same time, its sync checks passed (gate H2).
- The interference check between the reference device and the Ultraleap
  camera (protocol step 6).
- For the finger CMC, markers or sensors on each metacarpal (gate H3); the
  values would be the reference device's.
- An agreed acceptance threshold, if a mapping is to be accepted rather than
  reported.
- A second session on another day before any mapping is used beyond its own
  session.

## 10. Paths to evidence

`results\replay_fidelity\reference_validation_20261024\` (README inside):
`observability\`, `example\` (tracker session, reference capture,
comparison), `batch\` (the dataset run with the mapped columns), `viewer\`
(the mapped block on the example, refused on a dataset take), `gate_run.log`.

## 11. What to check first

- The canonical convention (`tmc_direction`, `tmc_angles`): the signs are
  pinned by hand-written vectors in `test_align_canonical_directions_match_hand_written_vectors`
  and against `tmc_estimate` in `test_align_estimate_axes_match_tmc_estimate`.
- That the fit sees no held-out sample: `test_split_the_fit_never_sees_a_held_out_sample`
  spoils every held-out value by 20 deg and checks the fitted `A` is
  identical and the held-out bias is exactly -20.
- That nothing is interpolated across a discontinuity and every exclusion is
  counted: `test_gaps_every_exclusion_names_its_reason_and_is_counted`.
- That a mapping fills nothing outside its scope, in the viewer and in the
  exports (`test_mapping_is_not_applied_outside_its_scope`,
  `test_mapped_columns_fill_only_the_recording_inside_the_mapping_scope`).
- Known limits: the order of a reference's two angles cannot be checked from
  data (0.12 deg in `test_align_a_wrongly_declared_order_is_nearly_invisible_in_the_errors`);
  `b` is a fitted parameter that can absorb gain (19 deg from the tracker's
  axis in the example against 5 deg true); the sync taps' distance
  thresholds are untested on real hands; the identifiability conditions are
  chosen here, not accuracy thresholds.
