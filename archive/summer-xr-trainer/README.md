# XR Trainer / StretchSense glove pipeline
# by Nicholas Kim. All rights reserved.
Real-time pipeline: **glove → XR Trainer → OSC (UDP) → validate → parse → 3D viz → record → CSV export**.

26 joints per hand (OpenXR `XR_HAND_JOINT` layout), each with local position + quaternion, captured at ~60 Hz.

## Install (Windows / PowerShell)

```powershell
cd "C:\Users\nkim2\OneDrive\Desktop\xr trainer\summer-xr-trainer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If `Activate.ps1` is blocked, run once per machine:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Daily use (real glove)

With XR Trainer running and the glove streaming on `127.0.0.1:9002`:

**One-command record + playback + export** (recommended):
```powershell
.\scripts\record_and_export.ps1               # 10 s default, 5 Hz
.\scripts\record_and_export.ps1 -Duration 30  # 30 s
.\scripts\record_and_export.ps1 -Hz 10        # sample 10x/sec instead of 5
```
Produces `recordings\realglove_YYYYMMDD_HHMMSS.jsonl` and `.csv`.

**Sampling rate**: recordings are downsampled to **5 frames/sec by default**
(`--hz 5`). The glove still streams at ~60 Hz; only the saved frames are
throttled, and each one keeps its exact `wall_time` / `iso_time` / `timestamp`.
Pass `--hz 0` (or `-Hz 0`) to keep every frame.

**Record the pose set (guided, hands-free)** — for the finger-pose
feasibility tests:
```powershell
python scripts/record_poses.py                          # 6 poses x 3 takes x 5 s
python scripts/record_poses.py --poses fist,peace --takes 2 --duration 4
python scripts/record_poses.py --mock --takes 1 --duration 2 --prep 2   # dry run
```
Waits until valid glove packets arrive, then for each pose: announces it,
counts down with beeps, records, moves on — no keyboard needed while the
glove is on. Default poses: `open_palm, fist, index_point, thumbs_up,
peace, pinch`. Each take is its own file in `recordings\poses\`
(e.g. `fist_left_take2_20260701_154212.jsonl`) and every frame carries
`"pose"` / `"take"` fields. Ctrl+C keeps the takes recorded so far.

**Record a single labeled take**:
```powershell
python scripts/record.py --pose fist --take 2 --duration 5
```

**Convert pose recordings to the 21-keypoint layout** (MediaPipe /
Ultralytics standard, for comparison with image hand-pose datasets):
```powershell
python scripts/export_keypoints21.py recordings\poses
```
Writes `keypoints21_frames.csv` (every frame) and `keypoints21_summary.csv`
(one representative frame per pose x hand — the recorded frame closest to
the pose average, so bone lengths stay rigid). Wrist-centred world XYZ in
metres; the 26 -> 21 joint mapping lives in `src/xr_hand/keypoints21.py`.

**Pose separability report** (fingertip-extension table + leave-one-out
nearest-centroid classification over the recorded takes):
```powershell
python scripts/analyze_poses.py                    # print report
python scripts/analyze_poses.py recordings\poses --write   # also update REPORT.txt
```

**Record a session** (unlabeled; starts immediately, stop with Ctrl+C):
```powershell
python scripts/record.py
```
Saves to `recordings\session_<id>.jsonl` and prints the playback command when
you stop. Add `--mock` to record without the glove.

**Scrub a session like a video**:
```powershell
python scripts/playback.py recordings\session_<id>.jsonl
```
3D hand on the left, both hands' per-joint data (world positions) on the right.

- **Frame** slider (or left/right arrow keys): scrub one frame at a time.
- **Select** slider: pick a time window (e.g. 1:30 to 1:35). The readout shows
  the window in both elapsed (`m:ss.mmm`) and wall-clock time, plus frame count.
- **Play/Pause**: auto-advances within the selected window.
- **Export selection → Excel**: writes just the frames in the window to a
  formatted `.xlsx` next to the recording (named with the time range). It has a
  **Summary** sheet (source, window, frame count) and a **Joint data** sheet
  laid out as a tidy table — one row per frame × hand × joint with world XYZ +
  rotation quaternion, a frozen/filterable header, per-hand colour, and a
  `moved` flag marking the joints that moved into each frame.

The data panel highlights the joints that moved into each frame, per hand.

**Just the live viewer** (no recording):
```powershell
python scripts/run_osc.py
```

**Live viewer + record** (no auto-close, close the window when done):
```powershell
python scripts/run_osc.py --record recordings/my_session.jsonl
```

**Diagnose a stream** (raw packet log, no viewer):
```powershell
python scripts/run_osc.py --dump --no-viz
```

## Without the glove (mock pipeline)

```powershell
python scripts/run_mock.py                                  # animated mock viewer
python scripts/record.py --mock --duration 10               # write mock recording
python scripts/playback.py recordings/<file>.jsonl          # scrub any recording
python scripts/export.py recordings/<file>.jsonl            # JSONL → CSV
python scripts/test_faults.py                               # headless fault checks
```

## Tests

```powershell
pytest -q
```

## Layout

```
src/xr_hand/
  joints.py       26 joint names + bone connectivity + dataclasses
  parser.py       187 OSC args → HandFrame (XYZW quaternion, local positions)
  validator.py    per-packet checks (length/NaN/zero/quat) + StreamMonitor
  mock.py         hand-frame generator with toggleable fault modes
  receiver.py     OSC server (port 9002, /v1/animation/kinematic/all)
  recorder.py     JSONL record + load
  viz3d.py        matplotlib 3D viewer (cyan left, red right, palm fill)
  playback_viewer.py  scrubbable viewer: 3D hand + per-joint data panel
  kinematics.py   forward kinematics (local transforms → world positions)
  keypoints21.py  26 OpenXR joints → 21-keypoint (MediaPipe) mapping
scripts/
  run_osc.py            live viewer (real glove) + optional --record / --duration
  run_mock.py           live viewer (in-process mock)
  record.py             headless recorder (supports --mock, --pose/--take labels)
  record_poses.py       guided hands-free pose session (countdown + beeps per take)
  analyze_poses.py      pose separability report (extension table + LOO classification)
  playback.py           scrub a .jsonl session (slider + per-joint data panel)
  export.py             .jsonl → .csv (one row per frame, joint columns)
  test_faults.py        headless fault-injection sanity check
  record_and_export.ps1 the one-command pipeline (record + playback + export)
tests/
  test_pipeline.py      parser, validator, StreamMonitor
  test_recorder.py      record/load round-trip
```

## Wire format reference

XR Trainer sends to `127.0.0.1:9002` on address `/v1/animation/kinematic/all` with 187 args:

- **Header (5)**: `[tick_counter, frame_id, status, "Reality Glove (L/R)" str, "Reality Glove" str]`
- **Joints (26 × 7)**: per joint `[x, y, z, qx, qy, qz, qw]`
  - position is **parent-relative** (local space)
  - quaternion is **XYZW**

Hand side is extracted from the device-label string at arg `[3]`. Header string fields are normalised to `0.0` placeholders so the downstream validator/parser see numeric data only.

## CSV format

One row per frame. Columns:
```
wall_time, iso_time, timestamp, packet_counter, hand_side, frame_id, status,
pose, take,
PALM_x, PALM_y, PALM_z, PALM_qw, PALM_qx, PALM_qy, PALM_qz,
WRIST_x, WRIST_y, WRIST_z, WRIST_qw, WRIST_qx, WRIST_qy, WRIST_qz,
THUMB_METACARPAL_x, ... (26 joints × 7 columns)
```
