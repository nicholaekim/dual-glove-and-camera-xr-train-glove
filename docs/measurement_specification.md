# Measurement specification: the six coordinates the tracker does not give

Written 2026-10-24. The six are Paper 3's `T_TMC_fe`, `T_TMC_aa` (thumb
trapeziometacarpal flexion and abduction) and `I_CMC`, `M_CMC`, `R_CMC`,
`L_CMC` (finger carpometacarpal, the palm arc). Paper 3 is Cobos et al. 2009,
"Human Hand Descriptions and Gesture Recognition for Object Manipulation"
(`paper3 (3).pdf`; pages are its printed pages). This page says, for each,
what has to be measured, what the existing tracker gives, and how accuracy
would be checked. It keeps two things apart: what Paper 3 assumes, and what
this dataset (`grasp set 2026-10-08 organized`, Ultraleap Stereo IR 170)
supports.

The software that takes real reference measurements is
`leap_hand.reference_validation` (format and comparison:
`docs/reference_validation.md`); the live procedure is
`docs/validation_capture_protocol.md`. Nothing on this page has been measured
against a reference device: no such device, participant session or recording
exists on 2026-10-24.

## Paper 3's model assumptions

These are properties of the paper's model, measured by its authors with a
Cyberglove. They define what the coordinates mean; they are not
measurements of this hand.

- **24 DoF in D-H chains** (Figure 1 p.3, Table 1 p.4, Tables 2 and 3 p.5).
  Thumb: joint 1 `T_TMC_aa` (alpha = pi/2), joint 2 `T_TMC_fe` with the
  metacarpal length, then MCP and IP flexion about axes parallel to the TMC
  flexion axis. So the paper turns the TMC **abduction first, then flexion**.
  Fingers: joint 1 `θi,CMC` with the metacarpal length and alpha = pi/2,
  joint 2 MCP abduction (alpha = -pi/2), joint 3 MCP flexion: at zero MCP
  abduction the CMC axis is parallel to the MCP flexion axis, so the paper's
  CMC is a flexion of the metacarpal against the carpus, "the palm arc, the
  palm deformation when the hand is grasping a ball" (p.4).
- **No numeric frames and no zero.** The segment frames are a drawing
  (Figure 1); the paper gives no zero posture and no sign rule beyond
  movements "positive or negative with regard to a reference line" (p.6).
- **Ranges** (Table 5, p.12, from anatomy books): TMC flexion/extension 50
  to 90 deg with 15 deg hyperextension, TMC abduction/adduction 45 to 60
  deg; CMC 5 (index), 5 (middle), 10 (ring), 15 (little) deg.
- **Couplings** (Table 6 p.15, eq. 3 to 8 pp.13-14): for example
  `I_CMC ≈ M_CMC`, `M_CMC ≈ 1/2 R_CMC`, `R_CMC ≈ 2/3 L_CMC`,
  `T_TMC_fe ≈ 11/10 T_MCP_fe`. Model relations for rebuilding a grasp from
  fewer angles; never used here to fill a coordinate.
- **No angular accuracy.** Table 8 (p.17) gives fingertip position errors of
  simplified reconstructions (7.67 % circular, 5.90 % prismatic for 9 DoF);
  these are not angular-accuracy thresholds, and the paper gives none.

## What this dataset supports

- A tracker skeleton fitted to two infrared images: 26 joints per frame,
  positions and orientations, on the camera's own clock
  (`leap_hand.replay_core`). The dataset's own dictionary says every angle is
  "derived from the tracker's bone orientations ... not goniometer or glove
  readings".
- The palm is one rigid frame. The four finger metacarpals do not move
  against it (measured below), so no recording of this tracker contains the
  palm arc.
- The thumb metacarpal does move against the palm (measured below). Against
  a calibration posture it gives a tracker-relative pair, the unvalidated
  estimate `T_TMC_*_est_openhand` / `T_TMC_*_est_neutral`
  (`docs/tmc_estimate.md`). The trapezium is not tracked, there are no
  palpated landmarks, and no independent device recorded these sessions, so
  nothing ties that pair to Paper 3's or the ISB's TMC angles.

## Observability, measured on the dataset

`scripts/leap/reference_validation.py observability` over every recording
(gate V11 of `GATES_reference_validation.md`): for each metacarpal, the
largest angle of any selected frame's orientation against the tracker's PALM
joint from the recording's mean, then the largest of those over the dataset.

| Bone | Recordings | Largest departure in any recording | Median recording |
|---|---|---|---|
| thumb metacarpal | 167 | 40.25 deg | 4.63 deg (static takes 2.39, transitions 16.28) |
| index metacarpal | 167 | 0.000023 deg | 0.0000077 deg |
| middle metacarpal | 167 | 0.000021 deg | 0.0000077 deg |
| ring metacarpal | 167 | 0.000021 deg | 0.0000079 deg |
| little metacarpal | 167 | 0.000024 deg | 0.0000093 deg |

71,709 frames. The finger values are the rounding of the stored quaternions:
the finger metacarpals are rigid with the palm in every frame. Verdict:
**finger CMC unobservable by this tracker; thumb TMC observable** (the
tracker reports its motion; how accurately is unknown). The 0.1 deg line the
command uses to say "does not move" is chosen for this software, four
thousand times the largest finger value, not an accuracy threshold.

## Device integrations inspected

| Device in this repository | What it measures | Can it be the reference? |
|---|---|---|
| Ultraleap Stereo IR 170 (`src/leap_hand`) | the tracker itself: a fitted skeleton, rigid palm, model metacarpals | no: a second calculation from it is a software cross-check (`data_origin` / `role` `software_cross_check`) |
| StretchSense glove (`src/xr_hand`) | stretch-sensor joint angles in the glove's own model; thumb CMC flexion only, finger metacarpal splay a fixed 5/10/15 deg template (`xr_hand.joint_frames`), no palm-arc sensor | an independent sensor for thumb flexion at most (`role` `independent_sensor`, results `cross_check`), not an anatomical reference; no CMC |
| Webcam, MediaPipe (`src/cam_hand`) | 21 learned image landmarks; no carpal landmarks | no |

None of them measures the trapezium, or each metacarpal against the carpus.
A reference device has to be added (`docs/validation_capture_protocol.md`,
"Required equipment").

## Thumb TMC: the trapezium and the first metacarpal

The TMC joint is the saddle joint between the trapezium (a bone of the distal
carpal row) and the base of the first metacarpal. Flexion and extension move
the metacarpal roughly in the plane of the palm (across it toward the little
finger, and out to the radial side); abduction and adduction move it roughly
at a right angle to the palm (palmar abduction lifts it away from the palm).
The ISB recommendation (Wu et al. 2005, J Biomech 38:981-992) builds a
trapezium frame from trapezium landmarks (section 4.3.3, adapted from Cooney
et al. 1981), a first-metacarpal frame (4.3.4), and a joint coordinate system
with flexion about the trapezium's axis first, abduction about the floating
axis, axial rotation about the metacarpal's axis (4.4.1); its neutral is the
posture where the two frames coincide (`docs/tmc_calibration_protocol.md`
quotes these sections).

**The tracker's palm as a proxy for the trapezium.** The tracker has no
trapezium. Its PALM joint is one rigid frame fitted to the whole hand. Using
it for the carpus assumes the trapezium keeps a fixed orientation in that
frame. The trapezium is bound to the bases of the second and third
metacarpals, whose CMC joints barely move (Paper 3 gives them 5 deg each,
Table 5), so the assumption holds to within that motion plus whatever the
tracker's palm fit does when the ring and little metacarpals arch (it has no
joint for them, so its fitted palm may tilt). Neither is measured. The fixed
part of the proxy, a fixed rotation not known in advance from the tracker's palm
axes to the trapezium's anatomical axes, is exactly what the mapping fits
(`A` in `leap_hand.reference_validation`); the changing part shows up in the
held-out errors. The same holds for the long axis: LeapC's metacarpal axis is
a model axis, so the mapping fits the anatomical axis in the tracker's
metacarpal frame too (`b`).

### T_TMC_fe

- **Segments and axes:** proximal segment the trapezium (ISB frame of
  section 4.3.3), distal the first metacarpal (4.3.4). Flexion axis: in
  Paper 3 the second TMC axis, parallel to the thumb MCP and IP flexion axes
  (Table 3); in the ISB joint coordinate system the trapezium's axis e1. In
  the canonical convention of the software: about the palmar normal e2 of
  the proximal frame at the zero posture.
- **Reference posture:** Paper 3 gives none. Either the ISB neutral (the two
  segment frames coincident; needs trapezium landmarks) or the clinical zero
  (the thumb straight beside the index in the plane of the palm, the
  `tmc_neutral_v1` posture). Each reference capture declares which
  (`frames.thumb_tmc.zero.id`); the result is valid for that zero only.
- **Rotation order:** Paper 3 abduction then flexion (D-H joints 1 and 2);
  ISB flexion then abduction then axial rotation. The capture declares it
  (`rotation_order`, refused when missing). The comparison cannot detect a
  wrongly declared order (it moves the held-out errors by 0.12 deg in
  `tests/test_reference_validation.py`), so it must come from the device's
  documentation.
- **Signs and units:** canonical: positive when the metacarpal head moves
  toward the ulnar side across the palm, both hands; degrees. A device that
  counts the other way declares `radial` and is flipped on import; a sign
  that is neither word is refused, never guessed.
- **What the tracker supplies:** the thumb metacarpal's orientation against
  the rigid palm in every frame (moves up to 40.25 deg in a recording). From
  it, against a calibration posture, the unvalidated estimate
  `T_TMC_fe_est_openhand` / `T_TMC_fe_est_neutral`. Nothing anatomical.
- **Additional measurement required:** an independent device recording the
  same hand at the same time and measuring the first metacarpal against the
  trapezium or a carpal frame: optical motion capture with markers and a
  static calibration trial building ISB-style frames, or electromagnetic
  sensors on the first metacarpal and the carpus; a goniometer reading of
  each static hold as the minimum (clinical convention, holds only). Plus
  thumb-index sync taps, isolated flexion and abduction sweeps, static holds
  and at least two trials (`docs/validation_capture_protocol.md`).
- **How accuracy is checked:** the mapping (palm-to-trapezium rotation `A`,
  metacarpal axis `b`) is fitted on the calibration trials only; on the
  held-out trials the comparison reports signed bias, MAE, RMSE, absolute
  error p50/p90/p95/max, the gain error of the isolated flexion sweep and its
  cross-talk into abduction, the repeatability across trials, the counts and
  exclusions, the tested range, and how the errors move when the clock is
  shifted by the stated synchronization uncertainty. No threshold is applied:
  none is sourced or agreed, Paper 3 has none.

### T_TMC_aa

- **Segments and axes:** the same two segments. Abduction axis: in Paper 3
  the first TMC axis (Table 3); in the ISB system the floating axis e2,
  perpendicular to the trapezium's flexion axis and the metacarpal's long
  axis. Canonical: about e1 = e2 x e3 of the proximal frame (in the plane of
  the palm, across the metacarpal).
- **Reference posture:** as for `T_TMC_fe` (one declared zero for the pair).
- **Rotation order:** as for `T_TMC_fe`; with flexion and abduction both non
  zero the two orders give different pairs for one direction (for example
  30 / 20 abduction-first reads 31.57 / 17.23 flexion-first), which is why
  the order is part of the coordinate's definition.
- **Signs and units:** canonical: positive when the metacarpal head moves
  palmarly, out of the plane of the palm (palmar abduction), both hands;
  degrees; `dorsal` declared is flipped on import.
- **What the tracker supplies:** the same metacarpal orientation; the
  unvalidated `T_TMC_aa_est_openhand` / `T_TMC_aa_est_neutral`.
- **Additional measurement required:** the same device and capture as for
  `T_TMC_fe`: a direction of the metacarpal needs both angles, so the two are
  measured and compared together.
- **How accuracy is checked:** as for `T_TMC_fe`, with the isolated
  abduction sweep giving the abduction gain error and its cross-talk into
  flexion.

## Finger CMC: the rigid palm

The second and third CMC joints barely move; the fourth and fifth flex,
cupping the palm (Paper 3's ranges 5, 5, 10, 15 deg, Table 5). Paper 3 models
each as the first joint of the finger's chain, a flexion of the metacarpal
against the carpus about an axis parallel to the MCP flexion axis (Table 2).
The tracker has no such joint: its four metacarpals are fixed in its palm
frame in every frame of the dataset (0.000024 deg at most). The motion is
**unobservable** by this tracker, and no calibration can make it observable:
there is no tracker signal to map.

### I_CMC

- **Segments and axes:** proximal the carpus (the trapezoid and capitate
  under the second metacarpal), distal the second (index) metacarpal; flexion
  axis parallel to the index MCP flexion axis (Paper 3 Table 2, joint 1).
- **Reference posture:** Paper 3 gives none; a flat palm (the metacarpal
  heads in one plane) is the natural zero, declared by the capture.
- **Rotation order:** one angle in Paper 3 (no CMC abduction); a device that
  reports more is compared on the flexion alone.
- **Signs and units:** canonical: positive when the metacarpal head moves
  palmarly (the palm arching); degrees.
- **What the tracker supplies:** nothing that moves: unobservable (index
  metacarpal fixed in the palm frame, 0.000023 deg at most). Its wrist-frame
  "CMC" row is the forearm-to-palm bend; against the palm it is a constant of
  the hand model.
- **Additional measurement required:** the index metacarpal against the
  carpus by an independent device (markers on the metacarpal and a carpal
  or forearm cluster, or a sensor on the metacarpal). The values would then
  be that device's measurements; the tracker cannot be validated for them.
- **How accuracy is checked:** not applicable to this tracker. A capture
  with a CMC column is compared for the record: the report gives the
  tracker's spread against the reference's range and the verdict
  `unobservable`, fits nothing, and the coordinate stays empty.

### M_CMC

- **Segments and axes:** the carpus (capitate) and the third metacarpal;
  axis parallel to the middle MCP flexion axis.
- **Reference posture:** a flat palm, declared by the capture.
- **Rotation order:** one angle (flexion).
- **Signs and units:** positive palmar (arching); degrees.
- **What the tracker supplies:** nothing that moves: unobservable (0.000021
  deg at most).
- **Additional measurement required:** the third metacarpal against the
  carpus by an independent device.
- **How accuracy is checked:** not applicable to this tracker; verdict
  `unobservable`, no fit, the cell stays empty.

### R_CMC

- **Segments and axes:** the carpus (hamate) and the fourth metacarpal;
  axis parallel to the ring MCP flexion axis.
- **Reference posture:** a flat palm, declared by the capture.
- **Rotation order:** one angle (flexion).
- **Signs and units:** positive palmar (arching); degrees.
- **What the tracker supplies:** nothing that moves: unobservable (0.000021
  deg at most), although the ring CMC is one of the two that do move in a
  real hand.
- **Additional measurement required:** the fourth metacarpal against the
  carpus by an independent device; the `cmc_palm_arc` item of the capture
  protocol moves it.
- **How accuracy is checked:** not applicable to this tracker; verdict
  `unobservable` with the reference's range beside the tracker's zero
  spread, no fit, the cell stays empty.

### L_CMC

- **Segments and axes:** the carpus (hamate) and the fifth metacarpal; axis
  parallel to the little MCP flexion axis.
- **Reference posture:** a flat palm, declared by the capture.
- **Rotation order:** one angle (flexion).
- **Signs and units:** positive palmar (arching); degrees.
- **What the tracker supplies:** nothing that moves: unobservable (0.000024
  deg at most) for the joint with the largest range in Paper 3 (15 deg).
- **Additional measurement required:** the fifth metacarpal against the
  carpus by an independent device, moved by the `cmc_palm_arc` item.
- **How accuracy is checked:** not applicable to this tracker; verdict
  `unobservable`, no fit, the cell stays empty.

## What no calibration can do

- **A posture calibration fixes a zero, not a frame.** The open-hand and
  `tmc_neutral` calibrations are one rotation each. A change of reference
  rotation mixes the two TMC angles by pose-dependent amounts
  (`docs/tmc_estimate.md`), so a zero does not remove cross-talk. The
  mapping fit refuses a single posture or a single movement plane
  (`fit_mapping`: "a single reference posture fixes a zero, not the frame").
  On the synthetic example the neutral-posture estimate is off by 4.2 / 6.3
  deg MAE with cross-talk slopes 0.075 / 0.088, where the mapped pair is off
  by 0.9 / 0.6 deg: synthetic numbers that show the mechanism, not this hand.
- **No calibration recovers motion the tracker does not report.** The finger
  CMC stay `unavailable` for this tracker whatever is recorded alongside it.
- **Paper 3's couplings are not measurements.** They are never used to fill a
  coordinate, and its fingertip errors are not angular thresholds.

## Status on 2026-10-24

| Coordinate | The tracker gives | Status | Outstanding |
|---|---|---|---|
| `T_TMC_fe` | metacarpal orientation (moves) | unavailable under the Paper 3 name; unvalidated estimate under `*_est_*`; `T_TMC_fe_mapped` empty (no reference capture) | a reference device, a participant session, the capture and comparison |
| `T_TMC_aa` | the same | as above | as above |
| `I_CMC` | nothing that moves | unavailable, unobservable by this tracker | a device measuring the metacarpal against the carpus; the values would be that device's |
| `M_CMC` | nothing that moves | unavailable, unobservable | as above |
| `R_CMC` | nothing that moves | unavailable, unobservable | as above |
| `L_CMC` | nothing that moves | unavailable, unobservable | as above |
