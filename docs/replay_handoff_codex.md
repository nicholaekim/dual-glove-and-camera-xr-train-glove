# Handoff for Codex: verified replay of the recorded hand movements and the Paper 3 layer

Written 2026-10-23 by the implementer (Claude). Codex verifies independently;
nothing below claims that independent verification is complete. Every number
here was measured by the commands listed in section 2 on this machine; rerun
them before trusting any of it.

Repository: `C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer`
(run everything from this folder with `.venv\Scripts\python.exe`; there is no
`python` on PATH).
Dataset (read-only): `C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized`.
Paper 3: `C:\Users\nkim2\OneDrive\Desktop\paper3 (3).pdf` (the Downloads copies
`paper1 (3).pdf`, `paper2 (3).pdf`, `paper3 (3).pdf` are byte-identical to the
earlier copies; the text extraction used is
`C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\paper3_text.txt`).

## 1. What changed and where

New files, plus one paragraph added to `README.md`; no existing module,
script, test, recording or dataset file was modified. Pre-existing
uncommitted changes in the working tree
(`docs/pilot_hand_on_object.md`, `scripts/minister_demo.py`,
`tests/test_minister_demo.py`) belong to other work and were left alone.

| File | Role |
|---|---|
| `docs/replay_plan.md` | the implementation plan written before coding, with the API contract and the declared tolerances |
| `GATES_replay.md` | the acceptance ledger (unlazy format): 11 gates, each with its exact CHECK command |
| `src/leap_hand/replay_core.py` | loading and hand selection, the three clocks and the fallback policy, segments and events, forward kinematics two ways, fidelity numbers, hold/interpolate pose lookup, reference frames, the playback clock and the scheduling log |
| `src/leap_hand/paper3.py` | Paper 3 relationships with page citations, grasp-class mapping, the 24 + 8 angles per frame, residual statistics, the corrected D-H chain and the eq. 9-11 fingertip error |
| `src/leap_hand/replay_viewer.py` | the matplotlib viewer (built on the same stack as `xr_hand.playback_viewer`): skeleton, recorded-vs-reconstructed overlay, joint axes, timeline markers, HUD with provenance, Paper 3 panel |
| `scripts/leap/replay_view.py` | the viewer's command line, headless screenshots, timing log |
| `scripts/leap/replay_verify.py` | the batch fidelity run over the whole dataset, the scenario reports and their checker, the dataset-unchanged check |
| `tests/test_replay_core.py`, `tests/test_paper3.py`, `tests/test_replay_viewer.py`, `tests/test_replay_verify.py` | the tests listed in section 3 |
| `docs/replay_fidelity.md` | method, tolerances, measured errors, policies, Paper 3 references and limitations |
| `docs/replay_handoff_codex.md` | this file |
| `README.md` | a short pointer to the new scripts in the Ultraleap section |

Existing code reused unchanged: `xr_hand.kinematics` (FK, quaternion
helpers), `xr_hand.joints`, `xr_hand.joint_frames` (wrist frame, the 24
angles, the paper-order MCP pair), the look and widgets of
`xr_hand.playback_viewer` / `xr_hand.viz3d`.

## 2. Exact launch and verification commands

All from the repository root. One command per block.

Unit tests of the new work:

```bash
.venv\Scripts\python.exe -m pytest tests/test_replay_core.py tests/test_paper3.py tests/test_replay_viewer.py tests/test_replay_verify.py -q
```

The whole suite:

```bash
.venv\Scripts\python.exe -m pytest tests -q -x
```

Batch fidelity run over all 115 static takes and 52 transitions, plus the four
scenario reports (writes `results\replay_fidelity\`):

```bash
.venv\Scripts\python.exe scripts\leap\replay_verify.py --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized" --out results\replay_fidelity --all --scenarios
```

Check the scenario reports against the known facts:

```bash
.venv\Scripts\python.exe scripts\leap\replay_verify.py --check-scenarios results\replay_fidelity
```

Check that the dataset is unchanged against the forensic SHA-256 baseline:

```bash
.venv\Scripts\python.exe scripts\leap\replay_verify.py --check-dataset-unchanged "C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\baseline_sha256.json" "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized"
```

Headless screenshots of the viewer for the four scenarios (no window):

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized" --scenario-screenshots results\replay_fidelity\screens
```

The interactive viewer (opens a window; keys: space play/pause, left/right
step a frame, + and - speed, n/p next/previous event, r reference frame,
i hold/interpolate, o recorded overlay, a joint axes, s screenshot, q quit):

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized" --grasp p3_circular_power --take 1
```

The acceptance ledger, re-executing every runnable gate:

```bash
node "%USERPROFILE%\.claude\skills\unlazy\scripts\gate-check.mjs" --reverify --timeout 1800 GATES_replay.md
```

## 3. Tests run and their results

Run on 2026-10-23 with `.venv\Scripts\python.exe -m pytest <file> -q`:

| File | What it pins | Result |
|---|---|---|
| `tests/test_replay_core.py` | rigid transforms and quaternion composition against an in-test Rodrigues reference; left and right synthetic hands through `absolute_to_relative` and back; exact replay at source frames; `PlaybackClock` speed, pause, seek; hold vs interpolate; slerp midpoint; repeated, nonmonotonic and missing stamps with the visible fallback chain; id change, time gap and dropped frames; selection and exclusion counts; the two real recordings' known facts | 32 passed |
| `tests/test_paper3.py` | the coefficients 4/3, 3/2, 3/2, 2, 4/5, 6/5 as exact fractions; applicability per class; the six unavailable coordinates; residual statistics on a synthetic series; the printed D-H matrix fails orthogonality at Table 2's alpha = +-pi/2 and the corrected one passes; eq. 9-11 error is zero when the constraints hold exactly; `frame_angles` reproduces `joint_frames.dof24` | 44 passed |
| `tests/test_replay_viewer.py` | the viewer on the Agg backend: stepping lands on recorded frames, time-driven ticks with an injected clock at 1x and 2x, pause, interpolate inside a segment and held across the id change, overlay, reference labels, timeline markers equal the events, event jumps, screenshot and timing log, HUD and Paper 3 text; the script's headless path; the real-data scenario screenshots | 20 passed |
| `tests/test_replay_verify.py` | the batch script on a synthetic dataset directory (three static takes, one transition with an id change): output rows, exclusions, events, the summary line; pooled statistics are concatenated frames, not averaged recordings; `--check-scenarios` with passing and failing fixtures; `--check-dataset-unchanged` positive, negative (one byte appended) and empty-intersection controls; the real scenarios | 27 passed |

123 new tests in all. The implementers also ran each test file against
deliberately broken copies of its module (21 for the core, 17 for
`paper3`) and every mutant was caught; those scratch runs are not part of
the repository.

The whole suite is gate R5 of `GATES_replay.md` (`pytest tests -q -x`,
executed by the ledger checker; its evidence line in that file carries the
exit code and the output digest). A second full run on 2026-10-23 on an
idle machine, `pytest tests -q`, ended `1306 passed, 5 skipped in 1207.54s`
(the 5 skips are pre-existing hardware-only tests). Two pre-existing test files are
timing-sensitive real-time mocks (`tests/test_leap_hand.py::test_the_hud_beeps_cannot_stall_the_frame_loop`,
`tests/test_record_poses_protocol.py`) and failed during the implementers'
runs only while a second pytest process was competing for CPU; each passed
when rerun alone. If R5 fails on one of them, rerun it on an idle machine.

## 4. Dataset coverage and measured fidelity errors

From the batch run of 2026-10-23 (`results\replay_fidelity\summary.md` and
`fidelity.json`; details and tables in `docs/replay_fidelity.md`):

```
FIDELITY OK: static=115 transitions=52 frames=71709 within_tolerance=167/167
```

| | static | transitions | all |
|---|---:|---:|---:|
| recordings | 115 | 52 | 167 |
| JSONL lines | 69,761 | 20,809 | 90,570 |
| selected frames (metadata hand and hand ids, status != 0) | 51,620 | 20,089 | 71,709 |
| excluded: the other hand | 18,141 (43 takes) | 720 (5 transitions) | 18,861 |
| excluded: another hand id on the selected side | 0 | 0 | 0 |
| excluded: status 0 | 0 | 0 | 0 |
| hold frames (metadata static interval) | 20,596 | n/a | 20,596 |

No frame of the selected hand was skipped by the replay: the only frame ids
missing from the selected sequences are the 40 (static) and 679
(transitions) at the seven identity-change boundaries, which the tracker
never reported for that hand.

| Quantity | Declared | Measured worst | Where |
|---|---|---|---|
| FK position vs recorded `abs26`, per joint (gated) | 1e-6 m | 0.8602 um (mean 0.4802 um, the mean uniform rounding to 1 um predicts) | p2_s01_power_grasp take 3, frame 14891, RING_DISTAL |
| matrix chain vs quaternion chain, angle (gated) | 1e-6 deg | 8.4e-14 deg | p1_cylindrical take 1 |
| matrix chain vs quaternion chain, position (gated) | 1e-9 m | 1.1e-16 m | p2_s02_power_grasp_ball take 2 |
| bone -z axis vs recorded bone vector (reported) | 0.05 deg expectation | 0.0108 deg (mean 0.00125 deg) | p3_circular_precision_thumb_2_fingers take 2, LITTLE_DISTAL |
| `capture_time` vs `timestamp_us` interval (reported) | 0.5 ms | 0.326 ms (median 0.0019 ms) | transition p2_s07_directional_power_side take 4 |
| relative quaternion norm minus 1 | 1e-9 | 2.2e-16 | |
| writer delay `wall_time - capture_time` | none | 838.1 ms static, 864.6 ms transitions | thumb_2_fingers take 4; transition p2_s08 take 4 |
| longest camera-clock step | none | 2.356 s | transition p2_s07_directional_power_side take 3 |

Every one of the 167 recordings uses `timestamp_us` with no fallback.
Events: 7 boundaries, each both an `id_change` and a `gap` (1 static, 6
transitions); no `dropped_frames`, `invalid_status` or `clock_defect`
events. The batch takes about two minutes.

Playback scheduling, measured separately on the TkAgg viewer on an idle
machine (`results\replay_fidelity\timing_tk_p3_circular_power_take1_speed1.json`
and `_speed0.25.json`): at 1x, 56 ticks in 8 s, lateness mean 74.8 ms,
p95 95.7 ms, max 128.2 ms, 392 source frames passed without being drawn; at
0.25x, 84 ticks, mean 18.6 ms, p95 23.4 ms, max 30.4 ms, 85 skipped. The
viewer is time-driven, so the frame shown is always the right source frame
for the media time; it is the draw rate, about 7 frames per second at 1x,
that is the limitation.

Independent versus same-path comparisons:

- Independent: reconstructed positions (chained from the parent-relative
  `joints`) against the recorded `abs26` positions; the reconstructed bone
  -z axis against the recorded bone vector (orientation from the quaternion
  chain versus direction from recorded positions).
- Two code paths over the same data (a cross-check, not an independent
  measurement): the matrix chain (`forward_kinematics_full`) against the
  quaternion-only chain (`reconstruct_quaternion_chain`).
- Not verifiable from this dataset: world orientations as the tracker
  reported them (the recorder wrote only the parent-relative quaternions),
  and physical accuracy against the hand.

## 5. Timestamp fallback and discontinuity behaviour

Clock preference: `timestamp_us` (LeapC acquisition clock, integer
microseconds) then `capture_time` (the same instant on the wall clock,
rounded to 1e-6 s) then `wall_time` (the writer; stalls up to about 0.86 s in
this dataset) then frame index times the nominal period. A clock is valid
when present on every selected frame and strictly increasing. The first valid
clock is used; any other choice sets `Timeline.fallback = True`, writes the
reason into `Timeline.reason`, adds a `clock_defect` event, and the viewer
HUD prints it in the warning colour. In this dataset all 167 recordings use
`timestamp_us` with no fallback; the fallback paths are covered by the tests
in `tests/test_replay_core.py` with fixtures carrying repeated, nonmonotonic
and missing stamps.

Discontinuity: a segment ends after frame i when the hand id changes
(`id_change`) or the camera-clock interval exceeds the gap threshold
(`gap`, default 0.05 s, about 4.5 frame periods at 90 Hz; both events are
raised when both hold). A frame-id skip with a continuous id and clock is a
`dropped_frames` event inside the segment and is still interpolated. Status-0
lines of the selected hand are `invalid_status` events and excluded from the
pose sequence. Interpolation (`mode=interpolate`) lerps world positions and
slerps world quaternions only between frames of one segment; across a
boundary the earlier frame is held and the HUD and `DisplayedPose.note` say
so. The default mode is `hold`: source frames only, no interpolation, no
smoothing, no anatomical correction anywhere in the replay path.

Playback is time-driven: at each tick the displayed frame is the latest
source frame at or before the media time, so a late tick shows a later frame
rather than stretching time. The scheduling error is logged per tick
(intended versus reached media time, frames skipped between displayed
frames) and written to the timing log; it is a property of the viewer and
the machine, not of the recording's timestamps.

## 6. Paper 3 references and remaining assumptions

Cobos, Ferre, Sanchez Uran, Ortego, Aracil, "Human Hand Descriptions and
Gesture Recognition for Object Manipulation", Computer Methods in
Biomechanics and Biomedical Engineering, 2009 (`paper3 (3).pdf`, page numbers
as printed):

- Table 1, p. 4: the 24 joint names. Tables 2 and 3, p. 5: the D-H chains
  (abduction before flexion at the MCP and TMC). Equations 1 and 2, p. 5.
- The printed D-H matrix, p. 5: row 1 column 3 reads `sin(alpha) sin(alpha)`
  and row 2 column 3 `-sin(alpha) cos(alpha)`; the standard transform has
  `sin(alpha) sin(theta)` and `-sin(alpha) cos(theta)`. `paper3.dh_as_printed`
  reproduces the print, `paper3.dh` the standard form; the test shows the
  printed rotation block is not orthogonal at Table 2's own alpha = +-pi/2
  and the corrected one is. The corrected form is used. This is the one
  correction made to the paper.
- Table 5, p. 12: joint ranges, second column "Hyperextension" (not
  abduction). Shown as annotations only; no bound test is built from it
  because it gives no signed minimum/maximum.
- Equations 3, 4, 5, 7, p. 13: inter-finger coupling measured while one
  finger flexes alone; listed, never applied to grasp recordings. Equation 6,
  p. 13 (ring and little abduction similar "in most cases"): reported as
  informational.
- Equation 8, p. 14 (PIP about 3/2 DIP, "the most accepted intra-finger
  constraint"): the only relation applied to the unmapped grasps.
- Table 6, p. 15 (circular reconstruction): MCP_fe about 4/3 PIP, PIP about
  3/2 DIP per finger; thumb MCP about 4/5 IP; TMC_fe about 11/10 MCP_fe and
  the CMC chain (unavailable here); M_MCP_aa about 1/5 I_MCP_aa; R_MCP_aa
  about 1/2 L_MCP_aa.
- Table 7, p. 16 (prismatic, thumb/index/middle): thumb MCP about 6/5 IP;
  MCP_fe about 3/2 PIP; PIP about 2 DIP; ring and little keep Table 6
  (section 5.1, p. 16).
- Equations 9 to 11, pp. 16 to 17, and Table 8, p. 17: the fingertip relative
  error of the 9-DoF and 6-DoF descriptions (7.67 % / 5.90 % for 9 DoF).
  These are fingertip-position percentages, not angular thresholds.

Assumptions that remain:

- Grasp classes: only the seven `p3_*` grasps are mapped (four circular,
  three prismatic) because they were recorded under Paper 3's own labels.
  The six Paper 1 and ten Paper 2 grasps are `unmapped`: no object-shape
  record supports a Cutkosky circular/prismatic assignment.
- Angle coordinates: MCP flexion and abduction for the relationships use the
  paper-order pair (abduction first, as the paper's D-H chain), with the
  main XYZ-order value reported beside it. Anatomical alignment of either
  with a Cyberglove reading is approximate (`xr_hand.joint_frames`
  docstring, "WHERE THIS IS NOT THE PAPER'S MODEL").
- Six coordinates are unavailable and marked so everywhere: T_TMC_fe,
  T_TMC_aa (the tracker's thumb metacarpal axes are not the paper's),
  I_CMC, M_CMC, R_CMC, L_CMC (LeapC's palm is rigid). Nothing is
  substituted for them.
- The eq. 9 fingertip error is computed relative to each finger's base
  frame (the wrist-to-base offset u_i of eq. 1 is unknown), with CMC = 0 on
  both sides and the thumb unavailable; it divides by |rp| as printed, so it
  can exceed 100 % when the rebuilt finger curls toward its base.
- No pass/fail threshold is applied to any Paper 3 relationship; residuals
  are reported with their spread. No coefficient was tuned and no angle was
  clamped.

## 7. Known limitations and incomplete items

- World orientations were never recorded (only the parent-relative
  quaternions and the world positions), so orientation fidelity is checked
  through the 19 bone directions against the recorded positions and through
  the two-path chain cross-check. A bone's roll about its own axis and the
  WRIST and PALM orientations have no recorded reference.
- The plan's premise for the bone-direction expectation was wrong: the
  shortest recorded bone is 7.36 mm, not about 14 mm, so `abs26` rounding
  allows up to 0.0135 deg there (measured maximum 0.0108 deg, declared
  expectation 0.05 deg unchanged). Corrected in `docs/replay_plan.md`
  ("Corrections after measurement"), `docs/replay_fidelity.md` section 9 and
  the `why` text of `replay_core.TOLERANCES`.
- The matplotlib viewer draws about 7 frames per second at 1x on this
  machine (section 4). Full-speed play therefore shows about one source
  frame in seven; stepping, scrubbing and event jumps show every frame
  exactly. A faster renderer was out of scope; the existing viewer stack
  was reused on purpose.
- The interactive controls were exercised on the real TkAgg window by
  injecting key presses through the viewer's own key handler
  (`results\replay_fidelity\interactive_tk\interactive_session.json` and its
  screenshots); the on-screen buttons call the same methods but were not
  physically clicked.
- No skinned mesh is drawn (skeleton only), so there is no retargeting that
  could alter a pose.
- Eq. 8 duplicates Table 6's PIP/DIP relation for circular grasps; the
  outputs mark the eq. 8 rows `duplicate_of` and they should be counted
  once.
- The eq. 9-11 fingertip error is not comparable with the paper's Table 8
  (finger-base denominator, CMC = 0, four fingers, LeapC angles); it is
  reported with those caveats and no verdict.
- The metadata static intervals were cut on the writer clock; in
  p3_circular_precision_thumb_2_fingers take 4 the declared 2 s interval
  covers 1.178 s of camera time. Replay uses the camera clock; durations
  read from the metadata are affected. The dataset was not changed.
- Physical accuracy against the person's hand is not assessable from this
  dataset.
- Nothing is committed: all new files are untracked in the working tree,
  and `results\` is git-ignored.

## 8. Paths to machine-readable verification results

All under `C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\results\replay_fidelity\`
(git-ignored; regenerated by the commands in section 2):

| Path | Contents |
|---|---|
| `fidelity.json` | totals, worst values with their recordings, clock statistics, the declared tolerances, `recording_fidelity` of all 167 recordings with their Paper 3 summaries |
| `fidelity_per_recording.csv` | 167 rows: counts, exclusions, clock source and fallback, segments, events, residuals, verdict, class |
| `events.csv` | the 14 timeline events with frame ids and camera-clock steps |
| `paper3_residuals.csv`, `paper3_residuals.json` | 167 x 30 relation rows: status, reason, `duplicate_of`, statistics over all frames and hold frames, paper-order and main-order |
| `paper3_pooled_by_class.csv` | per class and kind, pooled over frames |
| `paper3_fingertip_error.csv` | eq. 9-11 per recording |
| `summary.md` | the human summary of the run |
| `scenarios\timestamp_anomaly.json`, `identity_change.json`, `circular.json`, `prismatic.json` | the four scenario reports checked by `--check-scenarios` |
| `screens\*.png`, `screens\timing_headless_agg.json` | 19 headless screenshots of the four scenarios and the Agg timing log |
| `timing_tk_p3_circular_power_take1_speed1.json`, `..._speed0.25.json` | TkAgg playback timing on an idle machine |
| `interactive_tk\interactive_session.json`, `interactive_tk\timing_interactive_tk.json`, `interactive_tk\*.png` | the driven interactive TkAgg session: every control, with the HUD state after each |

The acceptance ledger with its evidence lines: `GATES_replay.md`.

## 9. A reproducible example for each real-data scenario

`DATASET` below stands for
`"C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized"`.
Each scenario's JSON is written by the `--scenarios` run of section 2 and
checked by `--check-scenarios`; a single recording's fidelity can also be
rerun on its own with `replay_verify.py --dataset DATASET --static <grasp> <take> --out <dir>`.

(a) Ordinary circular grasp: p3_circular_power take 1 (left hand, hand id
157, 449 of 898 lines selected, the other 449 are the right hand; class
circular). Report `scenarios\circular.json`: 15
applied relations with statistics, the six coordinates T_TMC_fe, T_TMC_aa,
I_CMC, M_CMC, R_CMC, L_CMC marked unavailable, within tolerance.

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --dataset DATASET --grasp p3_circular_power --take 1
```

(b) Prismatic grasp: p3_prismatic_precision take 1 (left hand, hand id 254,
450 of 900 lines selected, the other 450 are the right hand; class
prismatic). Report
`scenarios\prismatic.json`: 11 applied relations (Table 7 for thumb, index,
middle; Table 6 for ring and little and the abduction rows), the same six
unavailable coordinates.

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --dataset DATASET --grasp p3_prismatic_precision --take 1
```

(c) The writer stall: p3_circular_precision_thumb_2_fingers take 4, frames
151749 to 151855 (107 frames, the take's metadata static interval). Report
`scenarios\timestamp_anomaly.json`, `window`: writer span 1.99933 s, camera
span (`timestamp_us`) 1.178128 s, `capture_time` span 1.17809 s, clock source
`timestamp_us` with no fallback, largest writer jump 0.80744 s between frames
151852 and 151853 while the camera clock advanced 10.809 ms. In the viewer,
step to frame_id 151853 (selected index 224): the HUD prints the stall line
and the stamps; screenshot
`screens\p3_circular_precision_thumb_2_fingers_take4_writer_stall.png`.

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --dataset DATASET --grasp p3_circular_precision_thumb_2_fingers --take 4 --start-frame 224
```

(d) The identity change: p2_s03_power_grasp_finger_side take 1 (hand ids
573 and 574). Report `scenarios\identity_change.json`: two segments (indices
0 to 60, frame ids 88907 to 88967, id 573; indices 61 to 408, frame ids 89008
to 89355, id 574), one `id_change` and one `gap` event at the same boundary
with a camera step of 0.455302 s and 40 frame ids skipped, and the midpoint
pose in interpolate mode (t = 0.89443 s) returned as `held` on frame 88967
with the note "discontinuity (id_change, gap) between frame_id 88967 and
89008: holding the earlier frame, not interpolated". In the viewer the
timeline shows the orange id-change mark and the red gap at index 61; press
`i` then scrub between frames 60 and 61 to see the held pose; screenshots
`screens\p2_s03_power_grasp_finger_side_take1_interpolate_gap.png` and
`interactive_tk\05_interpolate_mid_gap_held.png`.

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --dataset DATASET --grasp p2_s03_power_grasp_finger_side --take 1 --start-frame 60 --mode interpolate
```

## 10. Addendum of 2026-10-23: angle coverage audit, visible MCP angles, thumb TMC estimate

Supervisor requirement received after section 1 to 9 were written. Plan:
`docs/angle_coverage_plan.md`; ledger: `GATES_angle_coverage.md` (eight
gates, C1 to C8).

**New and changed files.** New: `src/leap_hand/angle_coverage.py`,
`src/leap_hand/tmc_estimate.py`, `scripts/leap/tmc_estimate_report.py`,
`tests/test_angle_coverage.py`, `tests/test_tmc_estimate.py`,
`docs/angle_coverage.md`, `docs/tmc_estimate.md`,
`docs/angle_coverage_plan.md`. Changed: `scripts/leap/replay_verify.py`
(three angle exports, the calibration used, `--check-angle-exports`),
`scripts/leap/replay_view.py` (loads a TMC calibration for the estimate
block; `--tmc-calibration PATH`, `--no-tmc`), `src/leap_hand/replay_viewer.py`
(a third text panel, "Angles of the displayed source frame", with a separate
"estimated" block), `tests/test_replay_viewer.py`, `tests/test_replay_verify.py`,
`docs/replay_fidelity.md` (output table and batch time).

**The joint-by-movement table** (`angle_coverage.COVERAGE`, 43 entries; the
table in `docs/angle_coverage.md` is rendered from the code and a test fails
if they differ; `angle_coverage.csv` in every batch run): for every joint,
flexion/extension and abduction/adduction, plus the wrist's two movements and
the twist of every bone, with the output column, reference frame, zero, sign,
method, note, requirement and the dataset's own column. Statuses:
`calculated` (the 18 Paper 3 coordinates the camera gives plus the eight
paper-order MCP values), `estimated` (the two thumb TMC estimate columns),
`unavailable` (T_TMC_fe, T_TMC_aa, I/M/R/L_CMC flexion, and the four CMC
abductions, each with the recording or calibration it needs), and
`outside_paper3` (thumb MCP and IP abduction, finger PIP and DIP sideways
components, wrist flexion and deviation, and 24 twists, all exported under
`_tracker` names). No missing value is filled with zero or a Paper 3
prediction: the six unavailable columns are present and empty in every
exported row, and the export check fails on a single non-empty cell.

**Finger MCP visibility.** `angles_per_frame.csv` (one row per selected
frame, 71,709) carries `{I,M,R,L}_MCP_{fe,aa}` in both orders
(`xyz`, flexion first, and `_paper_order`, abduction first) among its 68
angle columns; `angles_per_recording.csv` gives each recording's
representative-frame value and hold-frame mean/min/max; the viewer's angle
panel lists each finger's MCP twice, labelled "MCP fe (xyz, flexion first)"
and "MCP fe (paper order, abduction first)", with the status beside every
value. At the 115 representative frames all 2,990 calculated values agree
with the dataset's `joint_angles_24.csv` within 0.005 deg.

**Thumb TMC estimate** (`leap_hand.tmc_estimate`, `docs/tmc_estimate.md`).
Mapping: the thumb metacarpal's rotation against the PALM frame, referred to
the session's open-hand rest orientation and split in the paper's order
(abduction first about the rest frame's y, flexion about the new x, the
thumb's hinge axis); the leftover turn about the bone is reported as twist.
Calibration: the chordal mean over the pre-movement frames of the 42
transitions that have a `movement_start_time_s` (4,379 frames; per
transition first, then across transitions). Assumptions measured on the
real data: hinge axes parallel (metacarpal vs proximal median 1.06 deg, max
7.63; proximal vs distal median 1.24, max 7.48); rest orientation agreement
across transitions median 4.4 deg, p90 7.8, max 20.0 (the estimate's floor
uncertainty); the metacarpal moves against the palm (within-take spread up
to 40.2 deg); the open-hand zero and the palm-as-carpus assumption are not
measurable here and leave a pose-dependent, unmeasured difference from Paper
3's anatomical coordinates (corrected in section 11: not a fixed offset).
Signs pinned by synthetic tests for both hands. Everything about it stays in
`*_est_openhand` columns, a separate viewer block and its own report
(`results\replay_fidelity\tmc_estimate\`); `paper3.compare` keeps the Table
6 TMC relation `unavailable`. Hold-frame means over the static takes:
circular 20 takes fe 3.0 / 18.7 / 58.1 (min / median / max), aa -38.3 /
3.8 / 17.4; prismatic 15 takes fe -13.5 / 3.5 / 18.9, aa -14.7 / -1.4 /
10.2.

**Commands** (repository root; `DATASET` as in section 9):

```bash
.venv\Scripts\python.exe -m pytest tests/test_angle_coverage.py tests/test_tmc_estimate.py tests/test_replay_viewer.py tests/test_replay_verify.py -q
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_verify.py --dataset DATASET --out results\replay_fidelity --all --scenarios --check-angle-exports
```

```bash
.venv\Scripts\python.exe scripts\leap\tmc_estimate_report.py --dataset DATASET --out results\replay_fidelity\tmc_estimate
```

```bash
node "%USERPROFILE%\.claude\skills\unlazy\scripts\gate-check.mjs" --reverify --timeout 1800 GATES_angle_coverage.md
```

**Results (2026-10-23).** Tests: `test_angle_coverage.py` 18,
`test_tmc_estimate.py` 40, `test_replay_viewer.py` and `test_replay_verify.py`
now 67 together (the four files 125). The batch run over the real dataset
ends

```
FIDELITY OK: static=115 transitions=52 frames=71709 within_tolerance=167/167
ANGLE EXPORTS OK: rows=71709 mcp_columns=16 est_columns=2 unavailable_empty=6 est_filled=71709
```

in 160.5 s (the calibration itself takes 3.3 s), and the TMC report ends
`TMC ESTIMATE REPORT OK: transitions_used=42 recordings=167` in 44.9 s. The
batch's per-frame estimates agree with the standalone report's within
5e-5 deg on all 71,709 frames. New result files under
`results\replay_fidelity\`: `angle_coverage.csv` (43 rows),
`angles_per_frame.csv` (71,709 rows, 48 MB), `angles_per_recording.csv`
(167 rows), `tmc_calibration_used.json`, and the folder `tmc_estimate\`
(`calibration.json`, `assumption_checks.json`, `estimates_per_recording.csv`,
`estimates_per_frame.csv`, `estimates_by_class.csv`). The viewer was launched
on the real TkAgg window (1920 x 1080 display at 125 % scaling, 125 dpi,
figure 15.4 x 8.0 in) with the calibration loaded: the angle panel fits
without cut-off and the estimated block shows the values and provenance
(`results\replay_fidelity\interactive_tk\12_tk_angle_panel_writer_stall.png`,
`tk_geometry_check.json`). The gate ledgers' evidence lines
(`GATES_angle_coverage.md`, and `GATES_replay.md` re-verified after these
changes) record the checker's own runs.

**Limits of the addendum.** The estimate's zero is the recorded open hand,
not Paper 3's; its floor uncertainty is the rest-frame agreement (median
4.4 deg, max 20.0 deg) and its difference from the paper's coordinates is a
reference change that mixes the two angles pose-dependently (section 11). The viewer's angle panel needs the 15.5 x 8.6 inch
figure; at higher display scaling rows may wrap. Nothing is committed from
the addendum yet.

## 11. Corrections of 2026-10-24 after Codex's independent verification

Codex's report (its temporary folder; copied to
`results\replay_fidelity\codex_corrections_20261024\codex_evidence\`) found
four things to correct. Plan: `docs/codex_corrections_plan.md`. Ledger:
`GATES_codex_corrections.md` (K1 to K12; the checker's evidence lines record
its own runs). Every output of this correction is under
`results\replay_fidelity\codex_corrections_20261024\` (its `README.md` lists
each file); `results\replay_fidelity\` itself still holds the 2026-10-23 run.
Nothing is committed; HEAD is still c777d4c. Nothing below claims that
Codex's independent verification is complete.

**(1) Changed files and why.**

| File | Change |
|---|---|
| `src/leap_hand/replay_core.py` | finding 2: `load_recording` records where each status-0 line of the selected hand sits; `find_segments` takes `invalid_before=` and ends a segment there (no event of its own: the `invalid_status` event is the record; no `dropped_frames` event there); `pose_at` names `invalid_status` in the hold reason; invalid lines before the first or after the last frame say so in their detail and cut nothing |
| `src/leap_hand/replay_viewer.py` | finding 1: `TICK_MS` 15 to 1 (Tk's `after()` quantum, below); the 3D axes, the panel frames and the slider track stay in the blitted background and only the hand's artists, the panel bitmaps, the slider's bar / handle / value and the timeline cursor are redrawn (`_draw_dynamic`); one blit of the changed region; `_TextPanel` composes its bitmap from a glyph atlas with per-row caching and writes it straight into the Agg buffer; the timing log and HUD gain `rendered_updates`, `frames_shown`, `wall_duration_s`, `rendered_per_s`, `frames_shown_per_s` and `runs_detail`; finding 4: the estimated block is titled by the calibration kind and shows that kind's columns |
| `src/leap_hand/tmc_estimate.py` | finding 4: `Calibration.kind` ("openhand" / "neutral") and `Calibration.hand`; `keys_for(kind)`; `from_json` reads the kind, infers it for older files from the writers' provenance prefixes, refuses anything else with the fix spelled out; `estimate` names its outputs by the kind and refuses the other hand (`mismatch`); `calibrate` sets kind and hand from the transition rows. Finding 3: the module docstring's "WHAT A CHANGE OF REFERENCE DOES", A3 and UNCERTAINTY rewritten |
| `src/leap_hand/tmc_calibration_take.py` (another session's uncommitted file; touched because the task names it) | `calibrate` sets `kind="neutral"` and the hand; `KEYS` from `tmc_estimate.keys_for("neutral")`; `is_neutral` reads the kind; `estimate` checks the kind and returns `tmc_estimate.estimate`'s dict; docstring |
| `src/leap_hand/angle_coverage.py` | the two neutral-posture estimate entries (45 entries), `ESTIMATE_COLUMNS_BY_KIND`, four `ESTIMATE_COLUMNS`; the estimate notes rewritten (finding 3) |
| `scripts/leap/replay_view.py` | the provider carries the kind, applies `tmc_estimate.mismatch` (hand and session) and a kind-specific provenance row; the console prints the kind, the hand and the output names; help and docstring |
| `scripts/leap/replay_verify.py` | the run's kind fills its two columns and leaves the other kind's empty; `T_TMC_est_calibration` = `<kind>:<session>:<n>t/<frames>f`; `--check-angle-exports` requires one complete pair of one kind per filled row with a calibration naming that kind (`est_columns=4`), fails on two kinds in one file; summary and `tmc_calibration_used.json` carry the kind and the hand; hand or session mismatch is reported |
| `scripts/leap/tmc_estimate_report.py` | A3 and A5 texts (finding 3); its `calibration.json` carries kind and hand through `to_json` |
| `scripts/leap/replay_playback_check.py` (new) | repeated real-window launches at 1x; reports ticks, rendered updates, distinct frames shown, skipped frames, wall duration, rates and lateness per trial; `PLAYBACK OK` / `FAIL` against `--min-frames-per-s` |
| `tests/test_replay_core.py` | four invalid-tracking tests (short interval, consecutive lines, before the first and after the last frame, coincident with an id change) |
| `tests/test_tmc_estimate.py` | Codex's counterexample with the additive control; outputs named by kind; kind and hand through JSON, older files read or refused; the other hand refused; a scan of the docs and modules for the old claim |
| `tests/test_tmc_calibration_take.py` (another session's file) | asserts the kind and hand; the estimate dict equality under the neutral names |
| `tests/test_angle_coverage.py` | 45 entries, the four estimate columns, the notes without the old claim |
| `tests/test_replay_viewer.py` | a neutral calibration through the real CLI (hand-made, and one made by the real take loader on synthetic takes), older files read by provenance or refused, the other hand not applied, the rendering counters per run, a saved figure showing the hand and the atlas-rendered text |
| `tests/test_replay_verify.py` | a neutral `--tmc-calibration` fills the neutral columns only; mixed kinds fail the checker; the open-hand run fills its hand's rows (56 of 66 in the fixture, the left-hand take left empty) |
| `docs/tmc_estimate.md`, `docs/angle_coverage.md`, `docs/angle_coverage_plan.md`, `docs/tmc_calibration_protocol.md` (another session's file, one paragraph), `docs/replay_fidelity.md` (sections 6 and 7), this file, `README.md` | the wording of finding 3 everywhere it stood; the calibration kinds; the invalid-tracking rule; the playback measurements |

Not touched: `scripts/leap/record_poses.py`, `scripts/minister_demo.py`,
`tests/test_minister_demo.py`, `docs/pilot_hand_on_object.md` (other
sessions' work), the dataset, `results\replay_fidelity\`'s 2026-10-23 run,
and the earlier ledgers. One consequence for an earlier ledger:
`GATES_angle_coverage.md` C4 expects `est_columns=2`; the same command now
prints `est_columns=4` (four estimate columns), so K5 of the new ledger is
the current oracle for the angle exports.

**(2) Reproduction and verification commands** (repository root; `DATASET`
as in section 9).

The reproductions before the fix are kept as files (item 8). After the fix:

```bash
.venv\Scripts\python.exe -m pytest tests/test_replay_core.py tests/test_tmc_estimate.py tests/test_tmc_calibration_take.py tests/test_angle_coverage.py tests/test_replay_viewer.py tests/test_replay_verify.py -q
```

```bash
.venv\Scripts\python.exe results\replay_fidelity\codex_corrections_20261024\invalid_frame\check_codex_file.py
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_verify.py --dataset DATASET --out results\replay_fidelity\codex_corrections_20261024\batch --all --scenarios --check-angle-exports
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_verify.py --check-scenarios results\replay_fidelity\codex_corrections_20261024\batch
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_verify.py --check-dataset-unchanged "C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\baseline_sha256.json" DATASET
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_playback_check.py --dataset DATASET --grasp p3_circular_power --take 1 --trials 3 --duration 6 --min-frames-per-s 30 --out results\replay_fidelity\codex_corrections_20261024\timing\after
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --file results\replay_fidelity\codex_corrections_20261024\viewer_inspection\synthetic\loose_left_invalid.jsonl --hand left --mode interpolate --start-frame 9 --tmc-calibration results\replay_fidelity\codex_corrections_20261024\viewer_inspection\synthetic\20990101_000000_left\calibration.json
```

```bash
.venv\Scripts\python.exe -m pytest tests -q -x
```

The last two reproduce the neutral-kind routing and the invalid-tracking
hold in the real window (the synthetic file has status-0 lines at frame_id
110 and 111; press `i`, scrub between frames 9 and 10) and run the whole
suite. The scripts that produced `before.json` / `after.json` and the
inspection screenshots were scratch drivers of the implementer's session,
not repository files; what they measured is reproducible with the commands
above and the kept input files.

**(3) Tests and dataset coverage.** The six test files hold 199 tests
(replay core 36, estimate 46, calibration take 24, coverage 18, viewer 33,
batch 42); gates K1, K3 and K4 run them, K9 the whole suite. The batch over
the dataset (gate K5, written to `codex_corrections_20261024\batch\`) must
end with `FIDELITY OK: static=115 transitions=52 frames=71709
within_tolerance=167/167` and `ANGLE EXPORTS OK: rows=71709 mcp_columns=16
est_columns=4 unavailable_empty=6 est_filled=71709`: the reconstruction and
its tolerances are untouched by these corrections, the open-hand estimate
fills every row (the calibration's hand, left, is the dataset's hand), and
the two neutral columns are empty (no take recorded). The dataset has no
status-0 line of the selected hand, so finding 2 changes none of its
numbers; its four regression tests are synthetic.

**(4) Rendering, before and after** (section 7 of `docs/replay_fidelity.md`
has the stage table). Real TkAgg window, 1x, p3_circular_power take 1
(449 frames, 4.989 s), three launches each on an idle machine
(`timing\before\`, `timing\after\`), first play run:

| | Ticks | Rendered updates | Distinct frames shown | Frames skipped | Frames shown / s | Lateness mean / p95 / max ms |
|---|---:|---:|---:|---:|---:|---|
| before, trials 1 to 3 | 52 / 52 / 52 | 52 / 52 / 52 | 52 / 52 / 52 | 396 / 396 / 396 | 10.6 / 10.6 / 10.6 | 82.3 / 95.1 / 125.0; 82.6 / 100.8 / 130.4; 82.1 / 95.4 / 136.9 |
| after, trials 1 to 3 | 191 / 179 / 178 | 190 / 178 / 177 | 190 / 178 / 177 | 258 / 270 / 271 | 38.0 / 35.6 / 35.3 | 25.2 / 28.5 / 136.1; 26.9 / 32.3 / 168.9; 27.2 / 33.4 / 148.4 |

The after rows are the gate K8 run recorded by the ledger's checker
(`timing\after\`); an earlier run of the same command a few minutes before
gave 187 / 186 / 185 ticks and 37.4 / 37.2 / 36.9 frames shown per second.
Codex's own run (54 ticks, 394 skipped) is the before state. The causes,
measured: a 64 ms tick (text panels 30 ms, the whole 3D axes 14.5 ms of
which 13.9 ms were its static panes, grid, ticks and labels, the Tk blit of
the whole figure 10 ms, the slider axes 2.2 ms; reconstruction, fidelity,
angles and Paper 3 rows under 3 ms together), plus Tk's `after()` rounding
every interval above 1 ms up to the 15.6 ms Windows timer quantum (15.6 to
31 ms of idle after each draw with the old 15 ms interval). After: about
20 to 26 ms a tick (blit of the changed region 8.6 ms, panels about 4 to 6,
the rest under 3) and a 1 ms re-arm. The media clock is untouched (wall time
times speed); the lateness is measured against a 1 ms slot and is the
previous draw's duration; the max is the first tick of a run, which pays for
the full draw that makes the background. Every source frame remains
reachable by stepping (`displayed pose: recorded`), and the panels show the
displayed frame's own values (the caches are keyed by frame index; the
per-frame `frame_fidelity` is still computed for the HUD).

**(5) Invalid-frame handling.** `invalid_frame\before.json`: Codex's file
loads as one segment, `pose_at` at 10 ms in interpolate mode returns
`interpolated` (alpha 0.5, "interpolated across 1 dropped frame ids") with
the wrist 10.000 mm from frame 100. `invalid_frame\after.json`: two segments
(0..0 and 1..1, both hand id 54), the one `invalid_status` event (line 2,
frame_id 101) kept, no `dropped_frames` event, `pose_at` returns `held` on
frame 100 with the note "discontinuity (invalid_status) between frame_id 100
and 102: holding the earlier frame, not interpolated", the pose equal to
frame 100's. `check_codex_file.py` prints `INVALID FRAME HELD` (gate K2).
Hold mode was already correct. In the real window
(`viewer_inspection\16_held_at_invalid_tracking.png`) a synthetic file with
status-0 lines at frame_id 110 and 111 shows the same hold between frames
109 and 112 and interpolates inside the valid stretches
(`17_interpolated_valid_data.png`).

**(6) Calibration kind: routing and compatibility.**
`calibration_kind\before.json`: a tmc_neutral calibration made by the real
take loader (`tmc_calibration_take.calibrate` on two synthetic takes),
loaded by `replay_view.py --file ... --tmc-calibration`, displayed the
open-hand title and `T_TMC_fe_est_openhand +20.0 / aa +10.0`, and the
file carried no kind. `calibration_kind\after.json`: the file carries
`"kind": "neutral"` and `"hand": "left"`, the window shows "estimated
(tmc_neutral calibration take; not a validated Paper 3 coordinate)",
`T_TMC_fe_est_neutral +20.0`, `T_TMC_aa_est_neutral +10.0`, twist 0.0, the
provenance row "neutral posture take, 2 takes; agreement ...", and the
console prints "kind: neutral (tmc_neutral calibration take: a held
reference posture); hand left; outputs T_TMC_fe_est_neutral,
T_TMC_aa_est_neutral". The standalone `tmc_calibration_take.estimate` gives
the same numbers under the same names. The viewer's default calibration
(`results\replay_fidelity\tmc_estimate\calibration.json`, written before the
field existed) is read as open hand from its provenance's first words
(`viewer_inspection\inspection.json`: `kind_field_in_file` null, kind
openhand, hand not recorded), and the regenerated one in
`codex_corrections_20261024\tmc_estimate\calibration.json` carries
`"kind": "openhand"`, `"hand": "left"`. A file without a kind whose
provenance starts with neither writer's words is refused by the viewer and
the batch with the fix spelled out (tests). A calibration of the other hand
gives the block "not computed (the provider gave no value)" with the reason
"calibration of the left hand, this recording is of the right hand: not
applied" (inspection.json, tests); a calibration of another session is
refused as before. In the batch a neutral `--tmc-calibration` fills only
`*_est_neutral`, names the calibration `neutral:<session>:<takes>t/<frames>f`,
and `--check-angle-exports` fails on a row with both kinds, on a calibration
cell naming the other kind, and on a file whose rows carry two kinds
(tests).

**(7) Remaining anatomical limitations.** Validated thumb TMC flexion and
abduction (`T_TMC_fe`, `T_TMC_aa`) and the four finger CMC flexions stay
unavailable: empty in every export, never 0, never a tracker-frame value
under an anatomical name, never a Paper 3 prediction. The thumb TMC
estimates are tracker-relative coordinates (the metacarpal against the
tracker's palm, referred to the open hand or to a held posture, decomposed
abduction-first) under assumptions A1 to A5; their difference from the
paper's anatomical angles is a change of reference rotation, which after the
decomposition mixes the two angles by pose-dependent amounts (Codex's
counterexample: a reference at 30 / 20 makes a pose at 30 / 50 read
3.32575 / 25.70535, not 0 / 30; additive only for a pure abduction-axis
reference change), so no zero offset ties them to the paper's. What is
still required: an anatomical reference frame on the same hand (trapezium
and first-metacarpal landmarks, Cheze et al. 2008) and an independent
sensor over the postures used, to measure the mapping's scale and
cross-talk; for the finger CMC, a sensor that measures each metacarpal
against the carpus. The neutral-posture calibration take has a protocol and
a tested loader but no recording on the real camera yet, and it fixes a
zero posture and its repeatability, not that frame. The synthetic tests
establish the software's behaviour, not physical anatomical accuracy.

**(8) Machine-readable results and screenshots.** Under
`results\replay_fidelity\codex_corrections_20261024\` (see its README):
`codex_evidence\` (Codex's files), `timing\before\trial1..3.json`,
`timing\after\trial1..3.json` and `summary.json`, `timing\profile_before.json`,
`timing\profile_after.json`, `timing\tick_gaps_after.txt`,
`invalid_frame\{before,after}.json` and `check_codex_file.py`,
`angle_offset\{before,after}.json`, `calibration_kind\{before,after}.json`
and the `*_work\` inputs, `viewer_inspection\01..18_*.png` with
`inspection.json` (the HUD and angle-panel text after every step),
`batch\` (the gate K5 run), `tmc_estimate\` (the regenerated open-hand
calibration with its kind), `gate_run.log`. The ledger with the checker's
evidence lines: `GATES_codex_corrections.md`.

## 12. Correction of 2026-10-24 (later): a calibration with no known hand is never applied

Codex's recheck (`codex_evidence\recheck\codex-recheck-report.md`) passed
the four corrections of section 11 and left one P2 finding: a
`calibration.json` written before the `hand` field existed loads with
`hand = None`, `tmc_estimate.mismatch` skipped the hand check for it, and in
loose `--file` mode the viewer applied the default left-hand calibration to
a right-hand recording. Gates K13 to K15 of `GATES_codex_corrections.md`
cover this correction; nothing is committed.

**Changed files and why.**

| File | Change |
|---|---|
| `src/leap_hand/tmc_estimate.py` | `mismatch` returns the reason `HAND_UNKNOWN` when the calibration's hand is None, so `estimate`, `series` and `tmc_calibration_take.estimate` (which all go through `estimate`) refuse it: no public entry point applies a calibration of unknown hand, and the hand is never taken from the recording. `Calibration.from_json(d, hand=)` takes the user's declaration for a file that records none and refuses one that contradicts a recorded hand. `hand_from_transitions(calibration, dataset_root)` recovers the hand of an open-hand file from the dataset's own `metadata/pose_transitions.csv`: every `per_transition` entry must match exactly one row (by `to_grasp`, `take`, and the session when the table names sessions) and all matched rows must record one hand; anything else gives None with the reason. Docstrings |
| `scripts/leap/replay_view.py` | `--tmc-calibration-hand left\|right`; `resolve_hand`: the file's hand, else the declaration, else (with `--dataset`) the table, else unknown; the console line says which ("hand left (recorded in the file)", "(declared with --tmc-calibration-hand)", "(recovered from metadata/pose_transitions.csv: 42 of 42 transitions ...)", or "hand not recorded in the file (...): the estimate is not applied; declare it ... or regenerate the file"); the block then shows "not computed (the provider gave no value)" with the reason as its provenance row |
| `scripts/leap/replay_verify.py` | the same flag and resolution for `--tmc-calibration`; `tmc_calibration_used.json`'s `source` and `hand` say how the hand is known; rows of a recording the calibration cannot be applied to stay empty and are listed under "Not estimated (the calibration is of another session or hand, or its hand is not known)" |
| `tests/test_tmc_estimate.py` | the six required cases at the core: unknown hand refused for left and right recordings, both kinds, through `mismatch`, `estimate`, `series` and the take estimator; known left and right hands apply only to their own hand; `from_json` with a declared hand (fills, agrees, contradicts, invalid); `hand_from_transitions` recovers the fixture's left hand from its 3 transitions and returns None for another session, an unmatched entry, no entries, no table, or two hands; the fixtures `cal_with(...)` now declare their hand |
| `tests/test_replay_viewer.py` | the older left file (both kinds) on a right-hand and on a left-hand recording through the provider and the real CLI: refused; applied with `--tmc-calibration-hand left`; refused with `right`; a declaration against a typed file exits 1; known left and right calibrations of both kinds against left and right recordings; the default-file path recovers the hand from a table, takes a declaration, and reports a contradiction |
| `tests/test_replay_verify.py` | the batch with a hand-less open-hand file: no row estimated and the reason in `summary.md` and `tmc_calibration_used.json`; with `--tmc-calibration-hand right` the right-hand rows; a contradiction exits 2; a hand-less file made by `tmc_estimate.calibrate` on the fixture dataset is recovered as right from the table (56 of 66 rows); the earlier fixtures now declare their hand (so the left-hand take in the fixture is refused by a right-hand calibration, 32 of 42 rows) |
| `docs/tmc_estimate.md`, `docs/angle_coverage.md`, this file | the rule, the three ways a hand becomes known, the default file's case |

**Before and after** (`results\replay_fidelity\codex_corrections_20261024\calibration_hand\`).
`before.json`: the default file has no `hand` (and no `kind`) field, loads
as openhand with hand None, `mismatch` returns None for both hands, and the
provider on a right-hand loose recording (a mirrored synthetic right hand at
flexion 20 / abduction 10) returns `T_TMC_fe_est_openhand 64.6`,
`T_TMC_aa_est_openhand -7.9`, twist -175.3 with the ordinary "open hand, 42
transitions" provenance. `after.json`: `mismatch` returns `HAND_UNKNOWN` for
both hands, the provider returns None values with the reason "calibration
hand not recorded (a file written before the hand field existed): declare it
with --tmc-calibration-hand left\|right, let --dataset recover it from
metadata/pose_transitions.csv, or regenerate the file: not applied", and
`tmc_estimate.estimate` raises the same. `default_with_dataset_console.txt`:
the same default file through `replay_view.py --dataset ... --grasp
p3_circular_power --take 1` prints "hand left (recovered from
metadata/pose_transitions.csv: 42 of 42 transitions of session
20261001_101621_left are left-hand recordings)" and the block is filled
(gate K14); `default_on_right_loose_console.txt`: through `--file
loose_right.jsonl --hand right --tmc-calibration <default file>` it prints
"hand not recorded in the file (no --dataset to recover it from): the
estimate is not applied; declare it with --tmc-calibration-hand left\|right
or regenerate the file" and the block says not computed.

**Verification commands** (repository root; `DATASET` as in section 9).

```bash
.venv\Scripts\python.exe -m pytest tests/test_tmc_estimate.py tests/test_tmc_calibration_take.py tests/test_replay_viewer.py tests/test_replay_verify.py -q -k "hand or older or refused or declared or applied or default or kind"
```

```bash
.venv\Scripts\python.exe -m pytest tests/test_replay_core.py tests/test_paper3.py tests/test_replay_viewer.py tests/test_replay_verify.py tests/test_angle_coverage.py tests/test_tmc_estimate.py tests/test_tmc_calibration_take.py -q
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --dataset DATASET --grasp p3_circular_power --take 1 --headless --screenshot results\replay_fidelity\codex_corrections_20261024\calibration_hand\default_with_dataset.png
```

```bash
.venv\Scripts\python.exe scripts\leap\replay_view.py --file results\replay_fidelity\codex_corrections_20261024\calibration_hand\after_work\loose_right.jsonl --hand right --headless --screenshot results\replay_fidelity\codex_corrections_20261024\calibration_hand\default_on_right_loose.png --tmc-calibration results\replay_fidelity\tmc_estimate\calibration.json
```

```bash
.venv\Scripts\python.exe -m pytest tests -q -x -p no:cacheprovider
```

Results: the seven focused test files hold 250 tests (replay core 36,
Paper 3 44, viewer 37, batch 43, coverage 18, estimate 48, take 24) and
pass (the implementer's focused run left out the take file's 8 recorder-mock
tests, which the whole suite runs), the two viewer commands print the lines
quoted above, and gate K15 runs the whole suite (its evidence line in the
ledger records the checker's run).
The batch over the dataset is unchanged by this correction: its
calibration is computed from the transition rows and carries `hand left`,
so gate K5's result stands (`FIDELITY OK ... 167/167`, `est_filled=71709`).

**How older files are handled now.** A file without `kind`: read from the
writer's provenance prefix or refused (section 11). A file without `hand`:
never applied as it is. It is applied only once its hand is known, in one of
three ways, each named in the console, in `tmc_calibration_used.json` and
in the viewer's block: recorded in the file (every file written since
2026-10-24); declared by the user with `--tmc-calibration-hand left|right`
(refused when the file records another hand); or, for an open-hand file,
recovered from the dataset's `metadata/pose_transitions.csv` when the table
names every transition the file lists, of the file's session, and all of
them record one hand. The hand is never inferred from a file name or a
session-name suffix, a calibration is never mirrored, and the recording's
hand is never substituted. Typed left and right calibrations of both kinds
behave as before: applied to their own hand, refused for the other with the
reason, and routed to `*_est_openhand` or `*_est_neutral`.

**Remaining limitations.** Recovery from the table trusts that the file's
`per_transition` list is the one the calibration was computed from (the
session and the exact set of transitions are checked; the rest matrix is
not recomputed). A neutral-take file without a hand cannot be recovered
from the table (its takes are not transitions) and needs the declaration or
regeneration. Everything in section 11's item 7 about anatomical validity
still holds. Codex's independent verification of this correction has not
been done.
