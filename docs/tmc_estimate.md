# Thumb TMC estimate, a tracker-relative coordinate pair

**This is an estimate, not a validated Paper 3 coordinate.** The numbers are
the thumb metacarpal's rotation against the tracker's palm, referred to a
chosen reference orientation and decomposed in the paper's order
(abduction first, then flexion). Two references exist, and each has its own
columns so the two zeros never share one: the open hand the participant held
before each transition cue (`*_est_openhand`, this page's measured values)
and the `tmc_neutral` posture of a calibration take (`*_est_neutral`,
"A calibration take instead of the open hand" below; none recorded yet).
Paper 3 (Cobos et al. 2009) fixes no numeric zero for the TMC, and no zero
offset could tie these to the paper's `T_TMC_fe` / `T_TMC_aa` in any case:
a change of reference turns both angles by pose-dependent amounts
("What a change of reference does", corrected 2026-10-24). The validated
`T_TMC_fe` and `T_TMC_aa` cells stay empty (`unavailable`), and
`paper3.compare` keeps the Table 6 relation `T_TMC_fe ~ 11/10 T_MCP_fe`
`unavailable` (`tests/test_tmc_estimate.py::test_paper3_tmc_relation_stays_unavailable`).

Code: `src/leap_hand/tmc_estimate.py`. Report: `scripts/leap/tmc_estimate_report.py`.
Tests: `tests/test_tmc_estimate.py`. Plan: `docs/angle_coverage_plan.md`, section
"The thumb TMC estimate (tmc leaf)". Results:
`results\replay_fidelity\tmc_estimate\`. All measured values below are from
the run of 2026-10-23 over `grasp set 2026-10-08 organized` (session
`20261001_101621_left`, one left hand, 115 static takes, 52 transitions,
71,709 selected frames).

## Mapping

Paper 3's thumb chain (Table 3, p.5) turns the TMC first about an abduction
axis (joint 1, alpha = pi/2), then about a flexion axis parallel to the MCP
and IP flexion axes (joints 2 to 4, alpha = 0). The tracker's thumb bone
frames have parallel x (hinge) axes (A1 below), so the metacarpal's x axis
stands in for that flexion axis. With `R_rest` the metacarpal's orientation
against the palm in the open hand:

```
R_pm     = R_palm^T R_metacarpal                       palm_to_metacarpal(rot)
R_tmc    = R_rest^T R_pm
(fe, aa) = joint_frames.paper_order_angles(R_tmc, hand)
           R_tmc = Ry(beta) Rx(alpha) [Rz(gamma)],  fe = -alpha,  aa = s*beta
           s = hand_sign: +1 right, -1 left
R_part   = Ry(s*aa) Rx(-fe)                            abduction first, then flexion
residual = R_part^T R_tmc                              what is left: a turn about the bone (z)
twist    = -s * atan2(residual[1,0], residual[0,0])    degrees, the sign joint_angles gives every twist
residual_deg = the full rotation angle of residual
```

`R_part` reproduces the bone axis of `R_tmc` exactly, so the residual is a
pure turn about the bone and `residual_deg` = |twist| up to rounding. Both are
written so a reader can check this.

Output names: `T_TMC_fe_est_<kind>`, `T_TMC_aa_est_<kind>`,
`T_TMC_twist_est_<kind>`, `T_TMC_residual_est_<kind>_deg` (degrees), with
`<kind>` the calibration's kind, `openhand` or `neutral`
(`tmc_estimate.keys_for`). Each row also carries the calibration's provenance
text and its kind.

## What a change of reference does (corrected 2026-10-24)

Codex's independent verification found the earlier wording of this page
wrong: it said the estimate differs from the paper's coordinates by a fixed,
unmeasured amount per angle. It does not, in general.

`R_tmc = R_rest^T R_pm`. Another reference `R_ref` (Paper 3's anatomical
zero, the ISB first-CMC neutral, a `tmc_neutral` take) gives
`R_tmc' = D R_tmc` with the fixed rotation `D = R_ref^T R_rest`. After the
abduction-then-flexion decomposition this is an additive constant only when
`D` is a pure turn about the first axis of the decomposition (the abduction
axis, `Ry`): then the abduction shifts by that angle and the flexion is
unchanged. For any other `D` both components change by amounts that depend
on the pose (cross-talk). The counterexample, on the implemented order and
reproduced by `tests/test_tmc_estimate.py::test_a_reference_change_mixes_the_two_angles_pose_dependently`
and `results\replay_fidelity\codex_corrections_20261024\angle_offset\`:

| | flexion | abduction |
|---|---:|---:|
| the reference itself, in the old coordinates | 30 | 20 |
| a pose, in the old coordinates | 30 | 50 |
| the same pose relative to that reference | **3.32575** | **25.70535** |
| what two additive offsets would give | 0 | 30 |
| the would-be "offset" at the reference | 30 | 20 |
| the would-be "offset" at this pose | 26.674 | 24.295 |

A pure turn about the flexion axis is not additive either (the test checks
it). So the difference between these estimates and Paper 3's anatomical
`T_TMC_fe` / `T_TMC_aa` is pose-dependent and unknown; subtracting a zero
measured in one posture does not remove it elsewhere. Describing the outputs
honestly: they are tracker-relative coordinates under the declared
assumptions (A1 to A5 below) and the declared reference; they can differ from
anatomical coordinates by cross-talk between the two angles and by amounts
that vary with the posture. Only an anatomical reference frame on the same
hand and an independent sensor could measure that difference ("What a
validated TMC would need"). The maths of the mapping is unchanged by this
correction.

## Calibration

`R_rest` is the chordal mean (the SVD projection of the averaged matrices onto
SO(3)) of `R_palm^T R_metacarpal` over the open-hand frames of the session's
transitions. Open-hand frames are those whose timeline time is before
`movement_start_time_s - 0.1 s`. Transitions with a blank movement start are
skipped, and so is any window shorter than 20 frames. The mean is taken in two
steps: first one per transition, then the mean of those 42 means. Every
transition therefore weighs once, however long its window.

- Used: **42 transitions** (29 valid, 13 questionable), **4,379 frames**,
  41 to 298 per window. Skipped: the 10 rows with a blank movement start
  (all 10 are `invalid`). No window was under 20 frames.
- `R_rest`: xyz angles (-27.90, -41.30, -79.22) degrees
  (`R = Rx(a) Ry(b) Rz(c)`; full matrix in `calibration.json`).
- Time origins. `rec.timeline.t` counts from the first selected frame on the
  camera clock. `movement_start_time_s` counts from `first_frame_capture_time`
  on the capture clock. On all 42 transitions the first selected frame's
  capture_time equals `first_frame_capture_time` (0.000 ms). Over the windows,
  origin and clocks together differ by at most **0.18 ms**, against an
  11.1 ms frame period.

## Signs (pinned geometrically by the tests)

- **Positive flexion** moves the metacarpal's bone direction (its -z) toward
  the rest frame's -y, the metacarpal's own palmar side, as for every joint in
  the repository. With this session's rest frame the unit motion in the palm
  frame is (-0.74, -0.47, -0.49) on the left hand: toward the palm's -y
  (palmar), mostly across the palm toward the little finger, and distally.
- **Positive abduction** moves the bone toward the rest frame's thumb-side x
  axis: +x on a left hand, -x on a right hand, the side `joint_frames` calls
  radial for every bone. With this session's rest frame the unit motion is
  (0.14, -0.81, 0.57) on the left hand: mostly toward the palm's -y (palmar
  abduction: the thumb leaves the palm's plane on the palm side), toward the
  wrist, and slightly radial.

Both statements hold for the left hand and for its mirror image, a right hand
(`test_positive_abduction_moves_toward_the_rest_frames_thumb_side_x`,
`test_positive_flexion_moves_toward_the_palm`). The real-data test checks the
palm-frame components for the measured `R_rest`.

## Assumptions, with the measured values

| | Assumption | Measured (2026-10-23) |
|---|---|---|
| A1 | The thumb's hinge axes are parallel (Table 3: TMC flexion axis parallel to MCP and IP). | Over all 71,709 frames, angle between the x axes. Metacarpal vs proximal: median **1.06**, p95 4.59, max 7.63 deg. Proximal vs distal: median **1.24**, p95 4.25, max 7.48 deg. Both maxima are in p2_s03_power_grasp_finger_side take 1, the tracker-identity-change take. On p3_circular_power take 1 the medians are 0.47 and 2.15 deg. For contrast, metacarpal x vs palm x: median 78.5, p95 85.8, max 92.0 deg. That gap is why the wrist-frame `T_TMC` pair is not the paper's. |
| A2 | The open-hand rest orientation is stable across the session. | Angle of each transition's mean to `R_rest`: median **4.4**, p90 **7.8**, max **20.0** deg. The worst is p2_s05_fingertip_grasp_side_support take 4. Next come p3_prismatic_precision take 3 (9.2), p1_spherical take 4 (8.7) and p3_circular_precision_thumb_2_fingers take 5 (8.0). Within one window the frames are steady: median spread 0.05 to 2.19 deg per transition, max 8.9 deg (p1_cylindrical take 4). Synthetic check: a transition shifted by 15 deg shows as the worst, at 15 - atan2(sin 15, 2 + cos 15) = 10.0 deg. |
| A3 | The open hand is the zero. | Not measurable here. Paper 3's zero is a drawing (Figure 1, p.3), and Table 5 gives ranges with no zero. The estimate is relative to the recorded open hand; its difference from the paper's coordinates is a change of reference, which after the decomposition is pose-dependent, not a fixed amount per angle ("What a change of reference does"). |
| A4 | The tracker's thumb metacarpal is tracked, not a rigid part of its palm. | Per recording, the largest angle of a frame to the take's own mean `R_pm`: median 4.6 deg over 167 recordings. That is 2.4 for static takes and 16.3 for transitions, which contain the movement. The max is 40.2 deg (p1_cylindrical take 3), with p1_lateral take 1 at 34.8. The take mean is 17.7 deg from `R_rest` at the median, 65.4 deg at most (p3_circular_precision_thumb_2_fingers take 1). The plan's pre-coding note said "up to 27 deg (p1_lateral take 1), 33 deg from the rest there". Here p1_lateral take 1 measures 34.8 and 33.2 deg; the dataset maximum is the higher 40.2 deg. Either way the metacarpal moves. |
| A5 | The palm frame stands for the carpus. | Not measurable here. The trapezium is not tracked and LeapC fits a rigid palm, so the TMC is read against the PALM joint's frame. The rotation from that frame to an anatomical trapezium frame is part of the pose-dependent reference difference of A3. |

Every number in this table is in `assumption_checks.json`: per-frame worst
locations, the per-transition and per-recording tables, and the time-origin
check.

## Uncertainty

- **Floor:** the across-transition agreement of the rest frame, median
  4.4 deg, p90 7.8 deg, max 20.0 deg. Any single estimate carries at least
  this much doubt about its zero.
- **On top of that, unknown:** the pose-dependent difference from Paper 3's
  anatomical coordinates (A3, A5: a change of reference with cross-talk
  between the two angles, not an offset to subtract), and the tracker's own
  error on the thumb. This dataset has no reference sensor to measure either.
- A1 adds a small model error. The hinge axes are 1 to 1.2 deg apart at the
  median and up to about 7.5 deg on the worst frames, so the flexion axis is
  not exactly the paper's.

## The estimate over the static takes, per Paper 3 class

Hold-frame mean per static take (the take's metadata static interval), then
min / median / max across the takes of each class (`paper3.grasp_class`).
Degrees, relative to the recorded open hand. From `estimates_by_class.csv`:

| Class | Static takes | `T_TMC_fe_est_openhand` min / median / max | `T_TMC_aa_est_openhand` min / median / max |
|---|---|---|---|
| circular | 20 | 3.03 / 18.68 / 58.14 | -38.26 / 3.83 / 17.36 |
| prismatic | 15 | -13.48 / 3.46 / 18.94 | -14.69 / -1.42 / 10.22 |
| unmapped (Paper 1 and 2 grasps) | 80 | -14.70 / 16.61 / 38.67 | -30.46 / -2.77 / 13.13 |

Per take, `estimates_per_recording.csv` gives the representative frame, the
hold-frame mean/min/max, the all-frame mean/min/max (twist included), the
residual max and the metacarpal motion. `estimates_per_frame.csv` has one row
per selected frame (71,709).

## How to regenerate

From the repository root:

```
.venv\Scripts\python.exe -m pytest tests/test_tmc_estimate.py -q
.venv\Scripts\python.exe scripts\leap\tmc_estimate_report.py --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized" --out results\replay_fidelity\tmc_estimate
```

The report ends with `TMC ESTIMATE REPORT OK: transitions_used=42
recordings=167` (2026-10-23: 44.9 s wall time). It counts both numbers
itself. The 42 is checked against its own count of the table's rows with a
movement start, and an `--out` inside the dataset is refused. Files written:
`calibration.json`, `assumption_checks.json`, `estimates_per_recording.csv`,
`estimates_per_frame.csv`, `estimates_by_class.csv`. Other consumers can load
the calibration with
`tmc_estimate.Calibration.from_json(json.load(open(".../calibration.json")))`.

## A calibration take instead of the open hand

Added 2026-10-23, not yet recorded on the camera: the protocol, the recorder
mode, the loader and their tests exist; every number on this page is still
from the open-hand zero.

`docs/tmc_calibration_protocol.md` records the thumb's neutral posture
(`tmc_neutral`: the hand flat with the palm to the camera, the fingers
straight and together, the thumb straight against the side of the index in
the plane of the palm, nothing pressed) held still 3 times for 5 s:
`scripts\leap\record_poses.py --calibration tmc_neutral --hand left`.
`leap_hand.tmc_calibration_take.calibrate(<session folder>)` turns the takes
into a `Calibration`: `R_rest` is the chordal mean of `R_palm^T R_metacarpal`
over every frame of each take (the file runs from the REC beep to the stop
beep), then the mean of the take means, so each take weighs once; each
take's hold is the span of the frames' own timestamps (at least 2 s and 20
frames, or the take is skipped with the reason); the across-take agreement
takes the place of the across-transition agreement as the zero's floor. The
recorder writes it as `calibration.json` in the session folder with
`"kind": "neutral"` and the hand, `Calibration.from_json` loads it, and
`estimate` runs against it unchanged: nothing in the mapping or the maths
above moves, only `R_rest`.

What changes is the zero. Against such a take the angles count from a stated
posture, the clinical reference position of the thumb (beside the index, in
the plane of the palm), instead of from the session's incidental open hand.
They are written under `T_TMC_fe_est_neutral`, `T_TMC_aa_est_neutral`,
`T_TMC_twist_est_neutral` and `T_TMC_residual_est_neutral_deg`, never under
`*_est_openhand`, so the two zeros never share a column. Since 2026-10-24
(Codex's finding 4) the kind is carried explicitly: `Calibration.kind` is
`"openhand"` or `"neutral"` and `Calibration.hand` the hand it was measured
on; both writers set them and `to_json` keeps them; `tmc_estimate.estimate`
names its outputs by the kind (`keys_for`), `tmc_calibration_take.estimate`
is that call with the kind checked first and refuses any other kind; the
viewer's estimated block is titled by the kind and shows that kind's rows
(`replay_view.py --tmc-calibration <session>\calibration.json`); the batch
(`replay_verify.py --tmc-calibration`) fills that kind's two columns of
`angles_per_frame.csv` and leaves the other kind's empty, with
`T_TMC_est_calibration` = `<kind>:<session>:<takes>t/<frames>f`; a
calibration of another hand or another session is not applied, with the
reason shown; and a `calibration.json` written before the field existed is
read by the first words of its provenance (`open-hand calibration` /
`tmc_neutral calibration`) or refused with the fix spelled out. Changing the
reference from the open hand to this posture is itself a reference change:
both angles move by pose-dependent amounts, not by a constant ("What a
change of reference does").

**A calibration with no known hand is never applied** (Codex's recheck of
2026-10-24). A `calibration.json` written before the `hand` field existed
(the 2026-10-23 open-hand file `results\replay_fidelity\tmc_estimate\calibration.json`
is one) loads with `hand = None`, and `tmc_estimate.mismatch` then refuses
every recording, left or right, so `estimate`, `series` and
`tmc_calibration_take.estimate` raise and the viewer's block and the batch's
columns stay empty with the reason "calibration hand not recorded ...". The
hand is never taken from the recording, from a file name or from a session
name. It becomes known in one of three ways, each named in the console and
in `tmc_calibration_used.json`: the file records it (every file written
since 2026-10-24); the user declares it with `--tmc-calibration-hand
left|right` on `replay_view.py` or `replay_verify.py` (refused when the
file records another hand); or, for an open-hand file, the dataset's own
`metadata/pose_transitions.csv` names every transition the file lists
(`per_transition`, matched by name, take and session) and all of them record
one hand (`tmc_estimate.hand_from_transitions`). The default file resolves
that way whenever `--dataset` is given: its 42 transitions are the 42
left-hand rows of session `20261001_101621_left`.

What it still cannot validate, without an independent sensor:

- **The posture is a stand-in, not the ISB neutral.** Wu et al. 2005 define
  the first CMC's neutral as the trapezium frame (Cooney et al. 1981) and the
  first-metacarpal frame coincident (section 4.4.1); the camera tracks no
  trapezium and no palpated landmark, so the offset between the held posture
  and that neutral is unknown (A3 and A5 in a new form), and Cheze et al.
  report that the rotation convention alone moves the three TMC angles by 9,
  2 and 13 degrees. A goniometer reading on the posture (optional in the
  protocol, kept in the provenance) measures the posture's own error in the
  clinical convention, not this offset.
- **The mapping's scale and cross-talk, and the tracker's error on the
  thumb.** Only a reference sensor on the same hand can measure them. Until
  one does, `T_TMC_fe` and `T_TMC_aa` stay `unavailable` under their Paper 3
  names, with or without a calibration take.

## What a validated TMC would need

From `docs/angle_coverage_plan.md`:

- **An anatomical reference frame**, not only a zero: because a change of
  reference mixes the two angles pose-dependently, tying the estimate to the
  paper's coordinates needs the full rotation from the tracker's palm and
  metacarpal frames to the anatomical trapezium and first-metacarpal frames.
  That means an ISB-style TMC frame from the trapezium and first-metacarpal
  landmarks (Cheze et al. 2008, as cited by Paper 3) measured on the same
  hand. A goniometer reading at a recorded reference pose, or the
  posture-based stand-in that now has a protocol
  (`docs/tmc_calibration_protocol.md`, above), fixes a zero posture and its
  repeatability, which is less than that frame; the landmark frame cannot
  come from this camera.
- **An independent sensor** to validate the mapping, as the paper used a
  Cyberglove. Its job is to measure the scale and cross-talk between the
  estimate's fe/aa and the true TMC angles over the range of postures used.

Until both exist, `T_TMC_fe` and `T_TMC_aa` stay `unavailable` under their
Paper 3 names, and the estimates stay under `*_est_openhand` and
`*_est_neutral`.

Both now have a software path (2026-10-24, `GATES_reference_validation.md`):
the specification of what has to be measured,
`docs/measurement_specification.md`; a file format for an independent
reference device and the comparison that fits the palm-to-trapezium rotation
and the metacarpal axis on calibration trials and reports bias, MAE, RMSE,
gain, cross-talk and repeatability on held-out trials,
`docs/reference_validation.md`; and the live capture,
`docs/validation_capture_protocol.md`. The comparison also evaluates this
estimate against the reference (`--tmc-calibration`), which is how its
cross-talk would be measured. A mapping's values appear as
`T_TMC_fe_mapped` / `T_TMC_aa_mapped` only inside the session, participant,
hand and tested range it was evaluated on. No reference capture exists yet.
