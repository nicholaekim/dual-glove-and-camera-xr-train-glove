# Correction plan after Codex's independent verification (2026-10-24)

Codex's report: `codex-verification-report.md` in its temporary evidence
folder (copied to `results\replay_fidelity\codex_corrections_20261024\codex_evidence\`).
Ledger: `GATES_codex_corrections.md` (K1 to K12). Every verification output of
this correction goes to `results\replay_fidelity\codex_corrections_20261024\`;
nothing in the dataset, in `results\replay_fidelity`'s earlier run or in the
other sessions' uncommitted files is overwritten.

What stays as it is: the reconstruction (167/167 recordings, 71,709 frames,
FK vs abs26 at most 0.8602 um), the camera-clock timeline, exact source-frame
stepping, the hand selection, the provenance texts, and the separation of
replay fidelity from the Paper 3 comparison.

## 1. Real-time playback (finding 1)

Reproduce first: three TkAgg trials of the current viewer at 1x on
p3_circular_power take 1 (449 frames, 4.989 s) with nothing else running
(`timing\before\`). Then profile one tick on the real path (reconstruction,
fidelity, angles, text panels, 3D draw, blit) before changing anything.
Fix the measured costs only: reuse the per-frame reconstructions the viewer
already holds, render only the text lines that changed, redraw only what moved,
blit only the dirty region. Count rendered updates (blits and full draws)
separately from timer ticks and skipped source frames; the timing log gains
`rendered_updates`, `wall_duration_s` and `rendered_per_s`. Acceptance: at
least 30 rendered updates per second in each of three trials
(`scripts\leap\replay_playback_check.py`, gate K9). The media clock is not
touched; every source frame stays reachable by stepping.

## 2. Interpolation through invalid tracking (finding 2)

A status-0 line of the selected hand that sits between two selected frames is
a discontinuity: `load_recording` passes those positions to `find_segments`,
which cuts the segment there; `pose_at` in interpolate mode then holds the
earlier frame and names `invalid_status` as the reason. The `invalid_status`
event with its line number and frame_id is kept as it was. Invalid lines
before the first or after the last selected frame cut nothing (there is
nothing to interpolate across). Regression tests: a short invalid interval,
consecutive invalid lines, invalid lines at the start and the end, the same
hand id on both sides, and continuous valid data still interpolating. Codex's
own synthetic file is re-run after the fix (gate K2).

## 3. Thumb-coordinate wording (finding 3)

The estimate is a tracker-relative coordinate pair: the thumb metacarpal's
rotation against the tracker's palm, referred to a chosen reference
orientation `R_rest` and decomposed abduction-first. Changing the reference
multiplies that rotation by a fixed rotation; after the decomposition this is
an additive constant only when the change is a pure turn about the first
(abduction) axis. In general both components change by posture-dependent
amounts (cross-talk), so the difference from Paper 3's anatomical TMC angles
is posture-dependent and unknown, not a zero offset to subtract. The maths is
not changed. Every sentence that claimed a fixed per-angle offset is replaced
(module docstrings, `angle_coverage.ESTIMATE_NOTE`, the report's text, the
docs, the coverage table, the handoff, the protocol page); Codex's
counterexample and a positive control (a pure abduction-axis reference change
is additive) become a test; a test scans the docs and code for the old claim.

## 4. Calibration kind (finding 4)

`tmc_estimate.Calibration` gains `kind` ("openhand" | "neutral") and `hand`.
Both writers set them; `to_json` carries them; `from_json` reads them, infers
`kind` from the two writers' own provenance prefixes for files written before
this change, and rejects anything else with the fix spelled out.
`tmc_estimate.estimate` names its outputs by the kind (`*_est_openhand` or
`*_est_neutral`); `tmc_calibration_take.estimate` keeps refusing a non-neutral
calibration. The viewer's estimated block, `replay_view.py`'s provider and
provenance line, and `replay_verify.py`'s exports and checker route by kind
(four estimate columns in the per-frame export, `est_columns=4`), and a
calibration of another session or another hand is not applied, with the
reason shown. The coverage table gets the two neutral-posture entries, so it
distinguishes calculated tracker coordinates, open-hand-relative estimates,
neutral-posture-relative estimates and unavailable anatomical coordinates.
Integration tests go through the real CLI/loading/provider path.

## Verification and handoff

Per issue: the reproduction before the fix (`codex_evidence\` and the
`before` files), the regression tests, the corrected behaviour, the existing
tests. Then the batch over the whole dataset with the angle-export check
(gate K6), the scenario check (K7), the dataset-unchanged check (K8), the
playback trials (K9), the full suite (K10), the real viewer inspected with
screenshots (K11) and the docs read in full (K12). The handoff for Codex is
section 11 of `docs/replay_handoff_codex.md`.
