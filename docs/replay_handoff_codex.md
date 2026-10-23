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
