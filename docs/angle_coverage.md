# Angle coverage: every joint and movement, its column and its status

Written 2026-10-23 (coverage leaf of `docs/angle_coverage_plan.md`; gates C1,
C3, C4 and C8 of `GATES_angle_coverage.md`). For every joint of the hand,
flexion/extension and abduction/adduction, plus the wrist and the twist of
every bone, this page says which number the camera recording gives, under
which column name, measured from which frame, from which zero, with which
sign and by which method, and with one of four statuses. Where the recording
gives no number it says what recording or calibration would.

Code: `src/leap_hand/angle_coverage.py` (`COVERAGE`, `all_angles`,
`write_coverage_csv`, `coverage_markdown`). Tests:
`tests/test_angle_coverage.py`. Exports: `scripts/leap/replay_verify.py`.
Viewer: the angle panel of `src/leap_hand/replay_viewer.py`.

## 1. Status vocabulary (one per entry)

| Status | Meaning |
|---|---|
| `calculated` | read off the tracker's bone orientations by a stated decomposition: the 18 Paper 3 coordinates the camera gives, plus the paper-order MCP pairs |
| `estimated` | calculated against a calibration the dataset itself supplies, with stated assumptions; kept in separate columns suffixed `_est_<calibration>` and in a separate viewer block, never merged with the validated coordinates (the thumb TMC pair, `docs/tmc_estimate.md`) |
| `unavailable` | a coordinate the tracker does not measure; the cell is empty; the entry's `requirement` says what recording or calibration would give it |
| `outside_paper3` | a movement the tracker reports but Paper 3's 24-DoF model does not have (thumb MCP and IP abduction, finger PIP and DIP sideways components, twist about the bone axis, wrist flexion and deviation); exported under a `_tracker` name so it is never mistaken for a Paper 3 coordinate (shown as "outside Paper 3" in the viewer) |

## 2. Frames, zero, signs and methods

- **Parent bone frame**: the parent joint's LeapC frame: x the hinge axis,
  y dorsal (the back of the hand), z along the bone toward the wrist.
  **Wrist frame**: the WRIST joint's frame, which on the camera is LeapC's
  forearm (`arm.rotation`, an estimate, not a tracked bone). **Palm frame**:
  the PALM joint's frame (the tracker's rigid hand).
- **Zero**: "bone in line with the parent" for a phalanx; "metacarpal along
  the reference frame's -z" for a metacarpal against the wrist or palm; "the
  session's open-hand pose" for the estimate; for the wrist, the PALM axes
  aligned with the WRIST (forearm) axes.
- **Signs, both hands**: flexion positive toward the palm; abduction positive
  toward the thumb side (radial); twist positive with the back of the bone
  turning toward the little finger; wrist flexion positive toward the palm
  side of the forearm, wrist deviation positive toward the thumb side
  (radial). The left/right mirror is taken out by `joint_frames.hand_sign`.
- **Methods**: `xyz` = R = Rx(a) Ry(b) Rz(c) of the parent-to-child rotation,
  flexion first (`joint_frames.joint_angles`: flexion = -a, abduction = b,
  twist = the third angle, signed by the hand); `paper_order` = abduction
  first, then flexion about the new x (`joint_frames.paper_order_angles`, the
  paper's D-H order, Table 2 p.5). The two orders read the same bone
  direction and differ once both angles are non-zero: sin(fe_paper) =
  sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe).
  `paper3.compare` takes the paper-order MCP pair as its primary value.

## 3. The table

The table below is the output of `leap_hand.angle_coverage.coverage_markdown()`
(one row per `coverage_rows()` entry), pasted unedited, so this page and the
code say the same thing; `tests/test_angle_coverage.py` fails if they differ.
To regenerate it after a change to `COVERAGE`:

```
.venv\Scripts\python.exe -c "from leap_hand.angle_coverage import coverage_markdown; print(coverage_markdown())"
```

43 entries: the thumb's 3 joints x 2 movements, the four fingers' 4 joints x
2, the wrist's 2 and one twist entry (41 joint-movement pairs), plus the two
`estimated` thumb TMC entries beside the two `unavailable` ones. The same
rows, with the same columns, are `angle_coverage.csv` of every batch run.

| Joint | Movement | Paper 3 name | Status | Output column(s) | Reference frame | Zero | Sign | Method | Note | Requirement | Dataset column |
|---|---|---|---|---|---|---|---|---|---|---|---|
| thumb TMC | flexion/extension | `T_TMC_fe` | unavailable |  | Paper 3's thumb frame: TMC flexion axis parallel to the MCP and IP flexion axes (Table 3, p.5) | Paper 3's thumb zero, given only as a drawing (Figure 1, p.3) | flexion positive toward the palm (both hands) | none: not measured | not measured as the paper defines it: the wrist-frame XYZ pair of the thumb metacarpal is about the hand's axes (the thumb's own flexion axis is about 81 deg from the palm's x), not the paper's TMC pair, and cannot be converted because the paper's thumb frame is a drawing (Figure 1, p.3); the dataset's cell is blank | a calibration recording with the thumb in an anatomically defined neutral (ISB-style TMC frame from the trapezium and first metacarpal landmarks, Cheze et al. 2008 as cited by Paper 3) or a goniometer reading at a recorded reference pose, so the estimate's zero can be tied to the paper's; plus an independent sensor (the paper used a Cyberglove) to validate the mapping | `T_TMC_fe` |
| thumb TMC | flexion/extension |  | estimated | `T_TMC_fe_est_openhand` | palm frame via the session's open-hand rest frame | the session's open-hand pose | flexion positive toward the palm (both hands) | paper_order on R_rest^T (R_palm^T R_metacarpal); see leap_hand.tmc_estimate | NOT a validated Paper 3 coordinate: relative to the recorded open hand, it differs from the paper's zero by an unknown constant per coordinate (Paper 3 gives its zero only as a drawing, Figure 1 p.3); the Table 6 relation T_TMC_fe ~ 11/10 T_MCP_fe stays unavailable | the session's open-hand frames: transitions with a movement_start_time_s, frames before movement_start_time_s - 0.1 s (leap_hand.tmc_estimate) |  |
| thumb TMC | abduction/adduction | `T_TMC_aa` | unavailable |  | Paper 3's thumb frame: TMC flexion axis parallel to the MCP and IP flexion axes (Table 3, p.5) | Paper 3's thumb zero, given only as a drawing (Figure 1, p.3) | abduction positive toward the thumb side, radial (both hands) | none: not measured | not measured as the paper defines it: the wrist-frame XYZ pair of the thumb metacarpal is about the hand's axes (the thumb's own flexion axis is about 81 deg from the palm's x), not the paper's TMC pair, and cannot be converted because the paper's thumb frame is a drawing (Figure 1, p.3); the dataset's cell is blank | a calibration recording with the thumb in an anatomically defined neutral (ISB-style TMC frame from the trapezium and first metacarpal landmarks, Cheze et al. 2008 as cited by Paper 3) or a goniometer reading at a recorded reference pose, so the estimate's zero can be tied to the paper's; plus an independent sensor (the paper used a Cyberglove) to validate the mapping | `T_TMC_aa` |
| thumb TMC | abduction/adduction |  | estimated | `T_TMC_aa_est_openhand` | palm frame via the session's open-hand rest frame | the session's open-hand pose | abduction positive toward the thumb side, radial (both hands) | paper_order on R_rest^T (R_palm^T R_metacarpal); see leap_hand.tmc_estimate | NOT a validated Paper 3 coordinate: relative to the recorded open hand, it differs from the paper's zero by an unknown constant per coordinate (Paper 3 gives its zero only as a drawing, Figure 1 p.3); the Table 6 relation T_TMC_fe ~ 11/10 T_MCP_fe stays unavailable | the session's open-hand frames: transitions with a movement_start_time_s, frames before movement_start_time_s - 0.1 s (leap_hand.tmc_estimate) |  |
| thumb MCP | flexion/extension | `T_MCP_fe` | calculated | `T_MCP_fe` | parent bone frame (THUMB_METACARPAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `T_MCP_fe` |
| thumb MCP | abduction/adduction |  | outside Paper 3 | `T_MCP_aa_tracker` | parent bone frame (THUMB_METACARPAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's thumb MCP has 1 DoF (Table 3, p.5) |  | `THUMB_PROXIMAL.abd_deg` |
| thumb IP | flexion/extension | `T_IP` | calculated | `T_IP` | parent bone frame (THUMB_PROXIMAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `T_IP` |
| thumb IP | abduction/adduction |  | outside Paper 3 | `T_IP_aa_tracker` | parent bone frame (THUMB_PROXIMAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's thumb IP has 1 DoF (Table 3, p.5) |  | `THUMB_DISTAL.abd_deg` |
| index CMC | flexion/extension (palm arc) | `I_CMC` | unavailable |  | the carpus (Paper 3's palm arc); the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | flexion positive toward the palm (both hands) | none: not measured | the wrist-frame metacarpal flexion is the forearm-to-palm bend, not the palm arc; against the palm it is a constant of the tracker's rigid hand | a sensor that measures each metacarpal against the carpus. LeapC fits a rigid palm; the StretchSense glove has no palm-arc sensor either. Options: optical markers on the four metacarpal heads and the wrist with a marker-based fit, or a tracker whose hand model has palm-arc joints. Without one, no recording of this setup can give it | `I_CMC` |
| index CMC | abduction/adduction |  | unavailable |  | the carpus; the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | abduction positive toward the thumb side, radial (both hands) | none: not measured | outside Paper 3's model and not measured: rigid palm | the same sensor as the CMC flexion (each metacarpal against the carpus); Paper 3's model has no CMC abduction, so even then it would be a tracker-only value outside the 24 |  |
| index MCP | flexion/extension | `I_MCP_fe` | calculated | `I_MCP_fe`, `I_MCP_fe_paper_order` | parent bone frame (INDEX_METACARPAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `I_MCP_fe; I_MCP_fe_paper_order` |
| index MCP | abduction/adduction | `I_MCP_aa` | calculated | `I_MCP_aa`, `I_MCP_aa_paper_order` | parent bone frame (INDEX_METACARPAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `I_MCP_aa; I_MCP_aa_paper_order` |
| index PIP | flexion/extension | `I_PIP` | calculated | `I_PIP` | parent bone frame (INDEX_PROXIMAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `I_PIP` |
| index PIP | abduction/adduction |  | outside Paper 3 | `I_PIP_aa_tracker` | parent bone frame (INDEX_PROXIMAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's index PIP has 1 DoF (Table 2, p.5) |  | `INDEX_INTERMEDIATE.abd_deg` |
| index DIP | flexion/extension | `I_DIP` | calculated | `I_DIP` | parent bone frame (INDEX_INTERMEDIATE) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `I_DIP` |
| index DIP | abduction/adduction |  | outside Paper 3 | `I_DIP_aa_tracker` | parent bone frame (INDEX_INTERMEDIATE) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's index DIP has 1 DoF (Table 2, p.5) |  | `INDEX_DISTAL.abd_deg` |
| middle CMC | flexion/extension (palm arc) | `M_CMC` | unavailable |  | the carpus (Paper 3's palm arc); the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | flexion positive toward the palm (both hands) | none: not measured | the wrist-frame metacarpal flexion is the forearm-to-palm bend, not the palm arc; against the palm it is a constant of the tracker's rigid hand | a sensor that measures each metacarpal against the carpus. LeapC fits a rigid palm; the StretchSense glove has no palm-arc sensor either. Options: optical markers on the four metacarpal heads and the wrist with a marker-based fit, or a tracker whose hand model has palm-arc joints. Without one, no recording of this setup can give it | `M_CMC` |
| middle CMC | abduction/adduction |  | unavailable |  | the carpus; the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | abduction positive toward the thumb side, radial (both hands) | none: not measured | outside Paper 3's model and not measured: rigid palm | the same sensor as the CMC flexion (each metacarpal against the carpus); Paper 3's model has no CMC abduction, so even then it would be a tracker-only value outside the 24 |  |
| middle MCP | flexion/extension | `M_MCP_fe` | calculated | `M_MCP_fe`, `M_MCP_fe_paper_order` | parent bone frame (MIDDLE_METACARPAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `M_MCP_fe; M_MCP_fe_paper_order` |
| middle MCP | abduction/adduction | `M_MCP_aa` | calculated | `M_MCP_aa`, `M_MCP_aa_paper_order` | parent bone frame (MIDDLE_METACARPAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `M_MCP_aa; M_MCP_aa_paper_order` |
| middle PIP | flexion/extension | `M_PIP` | calculated | `M_PIP` | parent bone frame (MIDDLE_PROXIMAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `M_PIP` |
| middle PIP | abduction/adduction |  | outside Paper 3 | `M_PIP_aa_tracker` | parent bone frame (MIDDLE_PROXIMAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's middle PIP has 1 DoF (Table 2, p.5) |  | `MIDDLE_INTERMEDIATE.abd_deg` |
| middle DIP | flexion/extension | `M_DIP` | calculated | `M_DIP` | parent bone frame (MIDDLE_INTERMEDIATE) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `M_DIP` |
| middle DIP | abduction/adduction |  | outside Paper 3 | `M_DIP_aa_tracker` | parent bone frame (MIDDLE_INTERMEDIATE) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's middle DIP has 1 DoF (Table 2, p.5) |  | `MIDDLE_DISTAL.abd_deg` |
| ring CMC | flexion/extension (palm arc) | `R_CMC` | unavailable |  | the carpus (Paper 3's palm arc); the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | flexion positive toward the palm (both hands) | none: not measured | the wrist-frame metacarpal flexion is the forearm-to-palm bend, not the palm arc; against the palm it is a constant of the tracker's rigid hand | a sensor that measures each metacarpal against the carpus. LeapC fits a rigid palm; the StretchSense glove has no palm-arc sensor either. Options: optical markers on the four metacarpal heads and the wrist with a marker-based fit, or a tracker whose hand model has palm-arc joints. Without one, no recording of this setup can give it | `R_CMC` |
| ring CMC | abduction/adduction |  | unavailable |  | the carpus; the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | abduction positive toward the thumb side, radial (both hands) | none: not measured | outside Paper 3's model and not measured: rigid palm | the same sensor as the CMC flexion (each metacarpal against the carpus); Paper 3's model has no CMC abduction, so even then it would be a tracker-only value outside the 24 |  |
| ring MCP | flexion/extension | `R_MCP_fe` | calculated | `R_MCP_fe`, `R_MCP_fe_paper_order` | parent bone frame (RING_METACARPAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `R_MCP_fe; R_MCP_fe_paper_order` |
| ring MCP | abduction/adduction | `R_MCP_aa` | calculated | `R_MCP_aa`, `R_MCP_aa_paper_order` | parent bone frame (RING_METACARPAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `R_MCP_aa; R_MCP_aa_paper_order` |
| ring PIP | flexion/extension | `R_PIP` | calculated | `R_PIP` | parent bone frame (RING_PROXIMAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `R_PIP` |
| ring PIP | abduction/adduction |  | outside Paper 3 | `R_PIP_aa_tracker` | parent bone frame (RING_PROXIMAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's ring PIP has 1 DoF (Table 2, p.5) |  | `RING_INTERMEDIATE.abd_deg` |
| ring DIP | flexion/extension | `R_DIP` | calculated | `R_DIP` | parent bone frame (RING_INTERMEDIATE) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `R_DIP` |
| ring DIP | abduction/adduction |  | outside Paper 3 | `R_DIP_aa_tracker` | parent bone frame (RING_INTERMEDIATE) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's ring DIP has 1 DoF (Table 2, p.5) |  | `RING_DISTAL.abd_deg` |
| little CMC | flexion/extension (palm arc) | `L_CMC` | unavailable |  | the carpus (Paper 3's palm arc); the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | flexion positive toward the palm (both hands) | none: not measured | the wrist-frame metacarpal flexion is the forearm-to-palm bend, not the palm arc; against the palm it is a constant of the tracker's rigid hand | a sensor that measures each metacarpal against the carpus. LeapC fits a rigid palm; the StretchSense glove has no palm-arc sensor either. Options: optical markers on the four metacarpal heads and the wrist with a marker-based fit, or a tracker whose hand model has palm-arc joints. Without one, no recording of this setup can give it | `L_CMC` |
| little CMC | abduction/adduction |  | unavailable |  | the carpus; the tracker's palm is one rigid frame | metacarpal along the reference frame's -z | abduction positive toward the thumb side, radial (both hands) | none: not measured | outside Paper 3's model and not measured: rigid palm | the same sensor as the CMC flexion (each metacarpal against the carpus); Paper 3's model has no CMC abduction, so even then it would be a tracker-only value outside the 24 |  |
| little MCP | flexion/extension | `L_MCP_fe` | calculated | `L_MCP_fe`, `L_MCP_fe_paper_order` | parent bone frame (LITTLE_METACARPAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `L_MCP_fe; L_MCP_fe_paper_order` |
| little MCP | abduction/adduction | `L_MCP_aa` | calculated | `L_MCP_aa`, `L_MCP_aa_paper_order` | parent bone frame (LITTLE_METACARPAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz (flexion first, joint_frames.joint_angles); paper_order (abduction first then flexion, the paper's D-H order, joint_frames.paper_order_angles) | two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the paper-order pair as its primary value |  | `L_MCP_aa; L_MCP_aa_paper_order` |
| little PIP | flexion/extension | `L_PIP` | calculated | `L_PIP` | parent bone frame (LITTLE_PROXIMAL) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `L_PIP` |
| little PIP | abduction/adduction |  | outside Paper 3 | `L_PIP_aa_tracker` | parent bone frame (LITTLE_PROXIMAL) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's little PIP has 1 DoF (Table 2, p.5) |  | `LITTLE_INTERMEDIATE.abd_deg` |
| little DIP | flexion/extension | `L_DIP` | calculated | `L_DIP` | parent bone frame (LITTLE_INTERMEDIATE) | bone in line with the parent | flexion positive toward the palm (both hands) | xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles) |  |  | `L_DIP` |
| little DIP | abduction/adduction |  | outside Paper 3 | `L_DIP_aa_tracker` | parent bone frame (LITTLE_INTERMEDIATE) | bone in line with the parent | abduction positive toward the thumb side, radial (both hands) | xyz second angle (joint_frames.joint_angles abd_deg) | the paper's little DIP has 1 DoF (Table 2, p.5) |  | `LITTLE_DISTAL.abd_deg` |
| wrist | flexion/extension |  | outside Paper 3 | `WRIST_fe_tracker` | WRIST frame (LeapC's forearm) to the PALM frame | PALM axes aligned with the WRIST (forearm) axes | flexion positive toward the palm side of the forearm (both hands) | xyz, from the PALM row of joint_frames.table_from_world (flex_deg) | not one of Paper 3's 24; the forearm is LeapC's estimate (arm.rotation), not a tracked bone |  | `PALM.flex_deg` |
| wrist | radial/ulnar deviation |  | outside Paper 3 | `WRIST_dev_tracker` | WRIST frame (LeapC's forearm) to the PALM frame | PALM axes aligned with the WRIST (forearm) axes | positive toward the thumb side, radial deviation (both hands) | xyz, from the PALM row of joint_frames.table_from_world (abd_deg) | not one of Paper 3's 24; the forearm is LeapC's estimate (arm.rotation), not a tracked bone |  | `PALM.abd_deg` |
| every bone (the 24 joints other than WRIST and PALM) | twist about the bone axis |  | outside Paper 3 | `THUMB_METACARPAL_twist_tracker`, `THUMB_PROXIMAL_twist_tracker`, `THUMB_DISTAL_twist_tracker`, `THUMB_TIP_twist_tracker`, `INDEX_METACARPAL_twist_tracker`, `INDEX_PROXIMAL_twist_tracker`, `INDEX_INTERMEDIATE_twist_tracker`, `INDEX_DISTAL_twist_tracker`, `INDEX_TIP_twist_tracker`, `MIDDLE_METACARPAL_twist_tracker`, `MIDDLE_PROXIMAL_twist_tracker`, `MIDDLE_INTERMEDIATE_twist_tracker`, `MIDDLE_DISTAL_twist_tracker`, `MIDDLE_TIP_twist_tracker`, `RING_METACARPAL_twist_tracker`, `RING_PROXIMAL_twist_tracker`, `RING_INTERMEDIATE_twist_tracker`, `RING_DISTAL_twist_tracker`, `RING_TIP_twist_tracker`, `LITTLE_METACARPAL_twist_tracker`, `LITTLE_PROXIMAL_twist_tracker`, `LITTLE_INTERMEDIATE_twist_tracker`, `LITTLE_DISTAL_twist_tracker`, `LITTLE_TIP_twist_tracker` | parent bone frame | no turn about the bone axis against the parent (third xyz angle 0) | positive with the back of the bone turning toward the little finger (both hands) | xyz third angle (joint_frames.joint_angles twist_deg) | Paper 3's bones have no rotation about their own axis; on LeapC's pure-swing finger joints this twist is -abd tan(flex/2), the order's, not the finger's (joint_frames, WHICH ORDER) |  | `<JOINT>.twist_deg` |

## 4. Nothing missing is filled

No missing value is filled with zero or with a Paper 3 prediction. The six
Paper 3 coordinates the camera does not give (`T_TMC_fe`, `T_TMC_aa`,
`I_CMC`, `M_CMC`, `R_CMC`, `L_CMC`; `joint_frames.CAMERA_NOT_GIVEN`) are
`None` in `all_angles`, empty in every row of `angles_per_frame.csv`
(`--check-angle-exports` fails on a single non-empty cell), named on one
"unavailable" line of the viewer's angle panel, and every Paper 3 relation
that needs one of them is `unavailable` in `paper3.compare`, with no numbers.
The Table 6 relations are never used to rebuild them.

One place in the code does put a number where a CMC would be, and it is not
an exported angle: the eq. 9-11 fingertip experiment of `leap_hand.paper3`
(`fingertip_reconstruction_error`) runs the paper's D-H finger chain with the
CMC joint left unrotated (0) in both the original and the rebuilt finger,
because the tracker's rigid palm has no CMC motion to give. That is a model
assumption of that experiment, stated in `paper3`'s module docstring ("EQ.
9-11 HERE") and in its `ERROR_NOTE`: the palm-arc part of the paper's error is
not in those numbers.

The thumb TMC estimate (`T_TMC_fe_est_openhand`, `T_TMC_aa_est_openhand`) is
not a fill either: it is a different coordinate (relative to the recorded
open hand, with an unknown offset from the paper's zero), in its own columns,
with its own status, computed by `leap_hand.tmc_estimate`
(`docs/tmc_estimate.md`); where it appears is in section 6, "Where the
estimate appears".

## 5. What each unavailable coordinate would need

- **Finger CMC (I, M, R, L; the palm arc)**: a sensor that measures each
  metacarpal against the carpus. LeapC fits a rigid palm; the StretchSense
  glove has no palm-arc sensor either. Options: optical markers on the four
  metacarpal heads and the wrist with a marker-based fit, or a tracker whose
  hand model has palm-arc joints. Without one, no recording of this setup can
  give it. (The wrist-frame metacarpal flexion is the forearm-to-palm bend,
  not the palm arc; against the palm it is a constant of the tracker's rigid
  hand.)
- **Finger CMC abduction**: not in Paper 3's model at all, and not measured
  (rigid palm). The same sensor as for the CMC flexion would measure it; it
  would still be a tracker-only value outside the 24.
- **Validated thumb TMC (`T_TMC_fe`, `T_TMC_aa`)**: a calibration recording
  with the thumb in an anatomically defined neutral (ISB-style TMC frame from
  the trapezium and first metacarpal landmarks, Cheze et al. 2008 as cited by
  Paper 3) or a goniometer reading at a recorded reference pose, so the
  estimate's zero can be tied to the paper's; plus an independent sensor (the
  paper used a Cyberglove) to validate the mapping. The wrist-frame XYZ pair
  of the thumb metacarpal is about the hand's axes (the thumb's own flexion
  axis is about 81 degrees from the palm's x) and cannot be converted,
  because the paper's thumb frame is given only as a drawing (Figure 1, p.3).

## 6. Where the numbers go

**`all_angles(pos, rot, hand_side)`** returns the 68 columns of
`ALL_ANGLE_COLUMNS` in table order: one `joint_frames.table_from_world` call
(wrist reference) gives the `_tracker` columns, and `paper3.frame_angles`
gives the 24 Paper 3 names (the six as `None`) and the eight paper-order MCP
values, so the Paper 3 numbers are exactly the ones `paper3.compare` is run
on. The `_tracker` columns are rows of that table: `T_MCP_aa_tracker` and
`T_IP_aa_tracker` the `abd_deg` of THUMB_PROXIMAL and THUMB_DISTAL,
`<i>_PIP_aa_tracker` and `<i>_DIP_aa_tracker` the `abd_deg` of each finger's
INTERMEDIATE and DISTAL, `WRIST_fe_tracker` and `WRIST_dev_tracker` the PALM
row's `flex_deg` and `abd_deg`, and `<JOINT>_twist_tracker` the `twist_deg`
of the 24 joints other than WRIST and PALM.

**`replay_verify.py`** (one batch run, `docs/replay_fidelity.md`) writes,
from the same reconstructed poses as the Paper 3 comparison:

| File | Rows | Columns |
|---|---|---|
| `angle_coverage.csv` | one per entry (43) | the table above (`coverage_rows()`) |
| `angles_per_frame.csv` | one per selected frame (71,709 over the dataset) | `kind, name, take, hand_side, hand_id, frame_index, frame_id, t_s, hold` (1/0 inside/outside the metadata's static interval, blank for transitions), the 68 columns of `ALL_ANGLE_COLUMNS` in table order (degrees, 6 decimals; the six unavailable empty), `T_TMC_fe_est_openhand`, `T_TMC_aa_est_openhand`, `T_TMC_est_calibration` (below, "Where the estimate appears") |
| `angles_per_recording.csv` | one per recording (167) | the representative frame's value (static takes: the frame whose `frame_id` is the metadata's `representative_frame_id`; transitions blank) and the hold-frame mean, min and max of each of the 26 `calculated` columns, the finger MCP in both orders |

`--check-angle-exports` (after the run, or alone with `--out`) checks the
per-frame file against `fidelity.json`: one row per selected frame, the 16
finger MCP columns (`{I,M,R,L}_MCP_{fe,aa}` and their `_paper_order` twins)
numeric in every row, the two estimate columns present, the six unavailable
columns empty in every row, and `angle_coverage.csv` with 43 rows.

**The viewer** (`scripts/leap/replay_view.py`) has a third text panel between
the HUD and the Paper 3 panel, "Angles of the displayed source frame
(degrees)": the sign line once; the wrist rows; per digit a line carrying the
twist of each of its bones, then each column with its value and status
(`calculated`, `outside Paper 3`); every finger MCP twice, labelled "MCP fe
(xyz, flexion first)" and "MCP fe (paper order, abduction first)", and the
same for aa; one line naming the six unavailable coordinates; and, apart, the
block "estimated (open-hand calibration; not a validated Paper 3
coordinate)", which says "not computed (no calibration loaded)" unless the
viewer is given a `tmc_provider`.

### Where the estimate appears

The thumb TMC estimate (`leap_hand.tmc_estimate`; mapping, calibration,
assumptions A1 to A5 and uncertainty in `docs/tmc_estimate.md`) is relative
to the recorded open hand, never a validated Paper 3 coordinate:
`T_TMC_fe` and `T_TMC_aa` stay empty everywhere.

- **`angles_per_frame.csv`**: `T_TMC_fe_est_openhand` and
  `T_TMC_aa_est_openhand` in every row the batch could estimate, and
  `T_TMC_est_calibration` naming the calibration as
  `openhand:<session>:<transitions>t/<frames>f`. The batch calibrates once
  per run (`tmc_estimate.calibrate` over the dataset's transitions, or
  `--tmc-calibration PATH`) and computes the estimate from the poses it has
  already reconstructed. 2026-10-23 run: all 71,709 rows,
  `openhand:20261001_101621_left:42t/4379f`; the values agree with
  `tmc_estimate_report.py`'s `estimates_per_frame.csv` to 5e-5 deg (that
  file's 4-decimal rounding) on every frame. Without a calibration (no
  transition with a movement start and 20 open-hand frames) the three
  columns are empty and `summary.md` says why; a recording whose metadata
  names another session than the calibration's is left empty too.
- **`angles_per_recording.csv`**: `T_TMC_fe_est_openhand_rep`, `_hold_mean`,
  `_hold_min`, `_hold_max`, the same for `T_TMC_aa_est_openhand`, and
  `T_TMC_est_calibration`.
- **`tmc_calibration_used.json`** (beside them): the calibration the run used
  (`Calibration.to_json`, plus `provenance_id`, `source`, `rows_estimated`),
  readable again with `--tmc-calibration`; or `"calibration": null` with the
  reason.
- **`--check-angle-exports`** appends `est_filled=<rows with both values>` to
  its OK line (`est_filled=71709` on the dataset) and fails on a row with one
  estimate value, or with values but no `T_TMC_est_calibration`.
- **The viewer's "estimated" block**: the two values, the twist about the
  bone with its residual, and one provenance row (transitions and the
  rest-frame agreement median/p90/max; the full provenance is printed when
  the calibration loads). `replay_view.py` with `--dataset` takes
  `--tmc-calibration PATH`, else
  `results\replay_fidelity\tmc_estimate\calibration.json` when it exists,
  else computes one (and prints how long it took); `--no-tmc`, or `--file`
  without `--tmc-calibration`, leaves it "not computed". The headless
  screenshots carry the same block.

## 7. The dataset's own columns, mapped

The organized dataset's `metadata/joint_angles_24.csv` and the readable
layer's `joint_measurements_readable.csv` (its `source_variable` column) come
from the same `xr_hand.joint_frames` table at each take's representative
frame:

- The 18 calculated Paper 3 names (`T_MCP_fe`, `T_IP`, `<i>_MCP_fe`,
  `<i>_MCP_aa`, `<i>_PIP`, `<i>_DIP`) and the eight `<i>_MCP_<k>_paper_order`
  columns are the columns of the same names here. Checked on the 2026-10-23
  run: at the 115 representative frames all 2,990 of them (26 columns x 115
  takes) in `angles_per_recording.csv` (`<column>_rep`) agree with
  `joint_angles_24.csv` to within 0.005 degrees, the dataset's rounding to
  0.01.
- `T_TMC_fe`, `T_TMC_aa`, `I_CMC`, `M_CMC`, `R_CMC`, `L_CMC` are blank in the
  dataset, as they are empty here (`unavailable`).
- `PALM.flex_deg` and `PALM.abd_deg` (the readable layer's wrist rows, read
  from the take's joint-frames file) are `WRIST_fe_tracker` and
  `WRIST_dev_tracker` (checked on five takes, to the file's 0.01). In the
  same joint-frames file (one row per joint at
  the representative frame) `<JOINT>.abd_deg` of THUMB_PROXIMAL,
  THUMB_DISTAL and each finger's INTERMEDIATE and DISTAL are the
  `_aa_tracker` columns and `<JOINT>.twist_deg` is `<JOINT>_twist_tracker`
  (the table's "Dataset column").
- `*_metacarpal_flex_wrist_ref`, `*_metacarpal_flex_palm_ref` and
  `T_metacarpal_abd_*` are what the tracker gives at the six unavailable
  joints, under names that say what they are (`docs/joint_frames.md`, "Where
  the 24 are not the paper's numbers"), with `forearm_to_palm_deg`, the wrist
  bend the wrist-reference ones include: the finger ones are the
  forearm-to-palm bend or the rigid palm's constants, the thumb ones the
  hand-axes pair. They are not Paper 3 coordinates and not in this table's
  exports.
