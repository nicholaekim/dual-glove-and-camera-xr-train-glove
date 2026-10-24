# Thumb TMC estimate, calibrated on the recorded open hand

**This is an estimate, not a validated Paper 3 coordinate.** The numbers are
the thumb metacarpal's movement against the tracker's palm, measured from the
open hand the participant held before each transition cue. Paper 3 (Cobos et
al. 2009) fixes no numeric zero for the TMC, so these differ from the paper's
`T_TMC_fe` / `T_TMC_aa` by an unknown constant per coordinate. They live only
in columns named `*_est_openhand`. The validated `T_TMC_fe` and `T_TMC_aa`
cells stay empty (`unavailable`), and `paper3.compare` keeps the Table 6
relation `T_TMC_fe ~ 11/10 T_MCP_fe` `unavailable`
(`tests/test_tmc_estimate.py::test_paper3_tmc_relation_stays_unavailable`).

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

Output names: `T_TMC_fe_est_openhand`, `T_TMC_aa_est_openhand`,
`T_TMC_twist_est_openhand`, `T_TMC_residual_est_openhand_deg` (degrees). Each
row also carries the calibration's provenance text.

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
| A3 | The open hand is the zero. | Not measurable here. Paper 3's zero is a drawing (Figure 1, p.3), and Table 5 gives ranges with no zero. The estimate is relative to the recorded open hand and is offset from the paper's by an unknown constant per coordinate. |
| A4 | The tracker's thumb metacarpal is tracked, not a rigid part of its palm. | Per recording, the largest angle of a frame to the take's own mean `R_pm`: median 4.6 deg over 167 recordings. That is 2.4 for static takes and 16.3 for transitions, which contain the movement. The max is 40.2 deg (p1_cylindrical take 3), with p1_lateral take 1 at 34.8. The take mean is 17.7 deg from `R_rest` at the median, 65.4 deg at most (p3_circular_precision_thumb_2_fingers take 1). The plan's pre-coding note said "up to 27 deg (p1_lateral take 1), 33 deg from the rest there". Here p1_lateral take 1 measures 34.8 and 33.2 deg; the dataset maximum is the higher 40.2 deg. Either way the metacarpal moves. |
| A5 | The palm frame stands for the carpus. | Not measurable here. The trapezium is not tracked and LeapC fits a rigid palm, so the TMC is read against the PALM joint's frame. The offset to an anatomical trapezium frame is part of A3's unknown constant. |

Every number in this table is in `assumption_checks.json`: per-frame worst
locations, the per-transition and per-recording tables, and the time-origin
check.

## Uncertainty

- **Floor:** the across-transition agreement of the rest frame, median
  4.4 deg, p90 7.8 deg, max 20.0 deg. Any single estimate carries at least
  this much doubt about its zero.
- **On top of that, unknown:** a constant offset per coordinate from Paper 3's
  zero (A3, A5), and the tracker's own error on the thumb. This dataset has no
  reference sensor to measure either.
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

## What a validated TMC would need

From `docs/angle_coverage_plan.md`:

- **A calibration recording** with the thumb in an anatomically defined
  neutral, so the estimate's zero can be tied to the paper's. That means an
  ISB-style TMC frame from the trapezium and first-metacarpal landmarks
  (Cheze et al. 2008, as cited by Paper 3), or a goniometer reading at a
  recorded reference pose.
- **An independent sensor** to validate the mapping, as the paper used a
  Cyberglove. Its job is to measure the scale and cross-talk between the
  estimate's fe/aa and the true TMC angles.

Until both exist, `T_TMC_fe` and `T_TMC_aa` stay `unavailable` under their
Paper 3 names, and this estimate stays under `*_est_openhand`.
