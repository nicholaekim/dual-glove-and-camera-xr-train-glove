# Angle coverage audit, MCP visibility and a thumb TMC estimate: plan

Written 2026-10-23 after the replay work (`docs/replay_plan.md`) and before
coding this addendum. Ledger: `GATES_angle_coverage.md`. Supervisor
requirement: audit angle coverage joint by movement, make the finger MCP
angles visible and labelled in the viewer and the exports, investigate a
defensible thumb TMC estimate with documented and tested calibration
assumptions kept apart from the validated Paper 3 coordinates, and say
precisely what the still-unavailable angles would need. Never fill a
missing value with zero or with a Paper 3 constraint prediction.

## Facts measured before coding (2026-10-23, read-only)

- Every recording of the dataset is one session, `20261001_101621_left`
  (115 static takes, 52 transitions), one left hand.
- 42 of the 52 transitions have a `movement_start_time_s`; their frames
  before `movement_start_time_s - 0.1 s` are the open hand before the cue
  (41 to 298 frames each). Over those frames the thumb metacarpal's
  orientation against the PALM joint is steady: median spread 0.1 to 2.2
  deg per transition (max 8.9 deg in p1_cylindrical take 4). Across the
  42 transitions the per-transition mean orientations agree with the
  session mean to a median of 4.4 deg, p90 7.8 deg, max 20.0 deg (the
  outlier is p2_s05_fingertip_grasp_side_support take 4).
- The thumb's three bone frames have parallel x axes (the hinge axes):
  metacarpal vs proximal median 0.5 deg (max 1.6), proximal vs distal
  median 2.2 deg (max 2.8), sampled over p3_circular_power take 1. This is
  the paper's Table 3 chain assumption (joints 2 to 4 have alpha = 0, so
  the TMC flexion axis is parallel to the MCP and IP axes).
- The thumb metacarpal's x axis is about 81 deg from the palm's x axis,
  which is why the wrist-frame XYZ pair the readable layer calls
  `T_TMC_fe/aa` is not the paper's TMC pair (`xr_hand.joint_frames`,
  "WHERE THIS IS NOT THE PAPER'S MODEL").
- LeapC's thumb metacarpal does move against the palm during grasps:
  within-take spread of the palm-to-metacarpal rotation up to 27 deg
  (p1_lateral take 1), and 33 deg from the open-hand rest there. The four
  finger metacarpals do not (rigid palm; the index metacarpal reads a
  constant -8.7, -9.6, -1.4 deg against the palm).

## What is built

| Leaf | Files | Role |
|---|---|---|
| coverage | `src/leap_hand/angle_coverage.py`, `tests/test_angle_coverage.py`, `docs/angle_coverage.md`, changes to `scripts/leap/replay_verify.py`, `src/leap_hand/replay_viewer.py`, `tests/test_replay_verify.py`, `tests/test_replay_viewer.py` | the joint-by-movement table as code and CSV; the full angle set per frame including the components outside Paper 3's model; `angles_per_frame.csv` and `angle_coverage.csv` exports; an angle panel in the viewer with the finger MCP values labelled by order and status |
| tmc | `src/leap_hand/tmc_estimate.py`, `tests/test_tmc_estimate.py`, `scripts/leap/tmc_estimate_report.py`, `docs/tmc_estimate.md` | the open-hand-calibrated thumb TMC estimate, its assumption checks on the real data, and its report |
| integration | `src/leap_hand/replay_viewer.py`, `scripts/leap/replay_verify.py`, `src/leap_hand/angle_coverage.py`, docs, `docs/replay_handoff_codex.md` | the estimate shown in the viewer's "estimated" section and written to the exports as `*_est_openhand` columns, never merged with the validated 18 |

## Status vocabulary (one per table entry)

- `calculated`: read off the tracker's bone orientations by a stated
  decomposition; the 18 Paper 3 coordinates the camera gives, plus the
  paper-order MCP pairs.
- `estimated`: calculated against a calibration the dataset itself
  supplies, with stated assumptions and an uncertainty; kept in separate
  columns suffixed `_est_<calibration>` and in a separate viewer section.
- `unavailable`: a Paper 3 coordinate the tracker does not measure; the
  cell is empty; the entry says what recording or calibration would give it.
- `outside_paper3`: a movement the tracker reports but Paper 3's 24-DoF
  model does not have (thumb MCP and IP abduction, finger PIP and DIP
  sideways components, twist about the bone axis, wrist flexion and
  deviation); exported under its own name so it is never mistaken for a
  Paper 3 coordinate.

## The table (joint x movement), as the coverage leaf must encode it

Reference frames: "parent bone frame" is the parent joint's LeapC frame
(x hinge, y dorsal, z along the bone toward the wrist); "wrist frame" is
the WRIST joint's frame (LeapC's forearm); "palm frame" is the PALM joint's
frame (rigid hand). Zero: "bone in line with the parent" for a phalanx;
"metacarpal along the reference frame's -z" for a metacarpal against the
wrist/palm; "the session's open-hand pose" for the estimate. Signs (both
hands): flexion positive toward the palm; abduction positive toward the
thumb side (radial); twist positive with the back of the bone turning
toward the little finger. Methods: `xyz` = R = Rx(a) Ry(b) Rz(c), flexion
first (`joint_frames.joint_angles`); `paper_order` = abduction first then
flexion (`joint_frames.paper_order_angles`, the paper's D-H order).

| Joint | Movement | Paper 3 name | Status | Output column(s) | Reference | Method |
|---|---|---|---|---|---|---|
| thumb TMC | flexion/extension | T_TMC_fe | unavailable (validated); estimated (open-hand calibration) | none validated; `T_TMC_fe_est_openhand` | palm frame via the session rest frame | paper_order on R_rest^T (R_palm^T R_metacarpal) |
| thumb TMC | abduction/adduction | T_TMC_aa | unavailable (validated); estimated | `T_TMC_aa_est_openhand` | as above | as above |
| thumb MCP | flexion/extension | T_MCP_fe | calculated | `T_MCP_fe` | parent bone (THUMB_METACARPAL) | xyz |
| thumb MCP | abduction/adduction | none (the paper's thumb MCP has 1 DoF) | outside_paper3 | `T_MCP_aa_tracker` | parent bone | xyz |
| thumb IP | flexion/extension | T_IP | calculated | `T_IP` | parent bone | xyz |
| thumb IP | abduction/adduction | none | outside_paper3 | `T_IP_aa_tracker` | parent bone | xyz |
| finger CMC (I, M, R, L) | flexion/extension (palm arc) | i_CMC | unavailable | none (the wrist-frame metacarpal flexion is the forearm-to-palm bend, not the palm arc; against the palm it is a constant of the tracker's rigid hand) | | |
| finger CMC | abduction/adduction | none | outside_paper3 and unavailable (rigid palm) | none | | |
| finger MCP | flexion/extension | i_MCP_fe | calculated | `i_MCP_fe` (xyz) and `i_MCP_fe_paper_order` | parent bone (metacarpal) | xyz; paper_order |
| finger MCP | abduction/adduction | i_MCP_aa | calculated | `i_MCP_aa` (xyz) and `i_MCP_aa_paper_order` | parent bone | xyz; paper_order |
| finger PIP | flexion/extension | i_PIP | calculated | `i_PIP` | parent bone (proximal) | xyz |
| finger PIP | abduction/adduction | none (1 DoF in the paper) | outside_paper3 | `i_PIP_aa_tracker` | parent bone | xyz |
| finger DIP | flexion/extension | i_DIP | calculated | `i_DIP` | parent bone (intermediate) | xyz |
| finger DIP | abduction/adduction | none | outside_paper3 | `i_DIP_aa_tracker` | parent bone | xyz |
| wrist | flexion/extension | none (not in the 24) | outside_paper3 | `WRIST_fe_tracker` | WRIST (forearm) to PALM | xyz, from the PALM row of `joint_frames.table_from_world` |
| wrist | radial/ulnar deviation | none | outside_paper3 | `WRIST_dev_tracker` | as above | xyz |
| every bone | twist about its axis | none | outside_paper3 | `<JOINT>_twist_tracker` for the 24 non-root rows | parent bone | xyz third angle |

`all_angles(pos, rot, hand_side)` in `angle_coverage` returns every output
column above that is calculated or outside_paper3 (None for the
unavailable ones), reusing `joint_frames.table_from_world` (one call) and
`paper3.frame_angles`. The `_tracker` suffix marks a value the tracker
gives but Paper 3 does not model; the `_est_openhand` suffix marks an
estimate.

## The thumb TMC estimate (tmc leaf)

Mapping. The paper's thumb chain (Table 3, p.5) turns the TMC about an
abduction axis first (joint 1, alpha = pi/2) and then about a flexion axis
parallel to the MCP and IP flexion axes (joints 2 to 4, alpha = 0). In the
tracker's frames the thumb bone frames' x axes are those parallel flexion
axes (measured parallel within 2.8 deg). So, with R_rest the thumb
metacarpal's orientation against the palm in the open hand:

    R_tmc = R_rest^T (R_palm^T R_metacarpal)
    (fe, aa) = paper_order_angles(R_tmc, hand)      # abduction first, then flexion about the new x
    twist   = the rotation left about the bone axis after Ry(aa) Rx(fe) is removed

Flexion positive toward the palm and abduction positive radial, by the
same signs as every other joint here; the leaf pins which way positive
abduction moves the metacarpal with a synthetic fixture and writes it in
the docstring.

Calibration. R_rest is the chordal mean over the open-hand frames of the
session's transitions (frames before `movement_start_time_s - 0.1 s`,
transitions with a blank start skipped, at least 20 frames), first per
transition and then across transitions. The calibration record carries the
number of transitions and frames, every per-transition spread, and the
across-transition agreement (median, p90, max), which is the estimate's
floor uncertainty.

Assumptions, each tested or measured and written in `docs/tmc_estimate.md`:

- A1 the thumb hinge axes are parallel (measured on every frame of the
  dataset: angle between the three x axes; expected a few degrees);
- A2 the open-hand rest orientation is stable across the session (measured
  across the 42 transitions; synthetic test with a deliberately shifted
  transition);
- A3 the open-hand pose is the zero. Paper 3 gives its zero only as a
  drawing (Figure 1, p.3) and Table 5's ranges fix no zero, so the estimate
  is "relative to the recorded open hand" and differs from the paper's by
  a change of reference (corrected 2026-10-24 after Codex's review: not a
  fixed amount per angle but a pose-dependent mix of the two,
  `docs/tmc_estimate.md`, "What a change of reference does"); it is never used as a validated
  Paper 3 coordinate and the Table 6 relation T_TMC_fe ~ 11/10 T_MCP_fe
  stays `unavailable`;
- A4 the tracker's thumb metacarpal is a tracked bone, not a rigid part of
  the palm (measured: within-take spread of R_palm^T R_metacarpal);
- A5 the palm frame stands for the carpus (the trapezium is not tracked).

The leaf's report writes `results\replay_fidelity\tmc_estimate\`:
`calibration.json`, `assumption_checks.json`, `estimates_per_recording.csv`
(representative frame, hold-frame mean, min and max per take, with the
calibration provenance), `estimates_per_frame.csv`.

## What the unavailable coordinates would need (to be written in docs/angle_coverage.md)

- Finger CMC (palm arc): a sensor that measures each metacarpal against the
  carpus. LeapC fits a rigid palm; the StretchSense glove has no palm-arc
  sensor either. Options: optical markers on the four metacarpal heads and
  the wrist with a marker-based fit, or a tracker whose hand model has
  palm-arc joints. Without one, no recording of this setup can give it.
- Validated thumb TMC: a calibration recording with the thumb in an
  anatomically defined neutral (ISB-style TMC frame from the trapezium and
  first metacarpal landmarks, Cheze et al. 2008 as cited by Paper 3) or a
  goniometer reading at a recorded reference pose, so the estimate's zero
  can be tied to the paper's; plus an independent sensor (the paper used a
  Cyberglove) to validate the mapping.

## Acceptance

`GATES_angle_coverage.md`: unit tests per leaf, the existing viewer and
verify tests, the batch exports with their row counts and the MCP columns,
the TMC report with the 42 usable transitions, the full suite, and two
manual gates (the viewer's angle and estimate sections inspected; the docs
and the handoff updated).
