# Camera hand tracking + glove/camera fusion
by Nicholas Kim. All rights reserved.

Webcam counterpart to the StretchSense glove pipeline in
`..\xr trainer\summer-xr-trainer`, plus the machinery to run both sensors at
once and fuse them.

**Why a camera at all.** The glove measures finger *flexion* directly and
keeps working when fingers hide behind the palm. It has no sensor for finger
*spread* or *thumb opposition* — which is exactly why `pinch` classified as
`open_palm` in the July results. A camera sees those directly. So this is not
a replacement for the glove; it is the other half of the hand.

| Degree of freedom | Measured by | Why |
|---|---|---|
| Finger curl (flexion) | **glove** | direct measurement, immune to occlusion |
| Finger spread (abduction) | **camera** | glove has no sensor for it |
| Thumb opposition | **camera** | the motion that makes `pinch` invisible to the glove |
| Hand position in space | **camera** | the glove reports nothing outside the wrist |

Landmarks are the 21-point MediaPipe layout — the same order
`xr_hand.keypoints21` already uses — so camera files, glove files and fused
files are directly comparable and export to the same formats.

## Install (Windows / PowerShell)

```powershell
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Then make the glove package importable from this venv (needed only by the
fusion scripts). Write one line — the path to the glove project's `src` — into
a `.pth` file in site-packages:

```powershell
"C:\Users\nkim2\OneDrive\Desktop\xr trainer\summer-xr-trainer\src" | Out-File -Encoding ascii .venv\Lib\site-packages\xr_hand_src.pth
```

Then fetch the hand-landmark model (a 7.5 MB MediaPipe binary, not kept in
this repo) into `models\`. Note `--ssl-no-revoke`: the same certificate
workaround the earlier vision experiment needed on this network.

```bash
curl.exe --ssl-no-revoke -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Check the install with `pytest -q` (expect 24 passed).

## Try it without any hardware

The whole pipeline runs on synthetic data, answer known in advance:

```powershell
python scripts/selftest_sync.py
python scripts/fuse_poses.py recordings/selftest
```

`selftest_sync.py` builds pose pairs that differ **only** in spread
(`spread_v` / `together_v`) or **only** in thumb opposition (`pinch` /
`relaxed`) — the two things a flexion sensor cannot see. Expected result, and
what it currently prints:

```
glove only      24/36 correct ( 67%)
camera only     36/36 correct (100%)
fused           36/36 correct (100%)
```

The glove misses exactly the spread- and opposition-defined poses. That
proves the code does what it claims on data whose answer is known; it does
**not** prove a real glove and webcam behave this way. A real session does.

## Daily use

**Live view** (start here — confirms camera, model and handedness):
```powershell
python scripts/live_view.py
```
Raise your right hand; the label should read `right`. If the labels read
backwards, add `--swap-hands` and say so, because it means the model's
convention changed (see `src/cam_hand/landmarks.py`).

**Record poses with the camera alone** — same protocol, poses and file naming
as the glove's `record_poses.py`:
```powershell
python scripts/record_poses_cam.py                          # 6 poses x 3 takes x 5 s
python scripts/record_poses_cam.py --poses pinch,fist --takes 2 --duration 4
```

**Record glove and camera simultaneously** — the data every fusion result
depends on. Needs XR Trainer streaming on `127.0.0.1:9002` *and* a webcam:
```powershell
python scripts/record_simultaneous.py
python scripts/record_simultaneous.py --mock-glove --takes 1 --duration 3 --prep 2   # rehearse, no hardware
```
Writes matched pairs into `recordings\sync\glove\` and `recordings\sync\cam\`
under one filename per take. Both recorders stamp `time.time()`, so the two
streams share a wall clock.

**Fuse and compare all three:**
```powershell
python scripts/fuse_poses.py                       # recordings/sync
python scripts/fuse_poses.py --write --export-csv results/fused.csv
```
Prints the glove-only / camera-only / fused table plus the pairing stats that
have to be right for the table to mean anything.

**Exports** (identical layouts to the glove pipeline's):
```powershell
python scripts/export_keypoints21_cam.py            # 21-keypoint CSVs
python scripts/export_prof_format_cam.py            # the professor's txt format
python scripts/analyze_poses_cam.py                 # camera separability report
```

**Score the camera against the professor's tracker** — his dataset has both
the photo and his device's keypoints for the same instant:
```powershell
python scripts/compare_to_tracker.py "..\xr trainer\xr trainer poses" --csv results/tracker_comparison.csv
```

## Results so far (102 reference frames)

Measured by `compare_to_tracker.py` against the professor's dataset:

- **Detection: 102/102 (100%).** MediaPipe found a hand in every frame,
  including fists and heavy occlusion.
- **Shape agreement: median 21 mm RMSE** over the frames where both devices
  reconstructed the same hand, after solving rotation + scale on the palm
  landmarks. Error concentrates in fingertips and the thumb (30–48 mm) and is
  small at the knuckles (3–12 mm) — the camera's weakness is depth on the
  parts that occlude, exactly as expected.
- **Chirality disagreement on 45/102 frames.** MediaPipe labels 98 of 102
  frames `left`; the tracker labels 55 `left` and 47 `right`, and every
  disagreement is a frame the tracker called `right`. On those frames the two
  3-D reconstructions are mirror images (mirrored refit drops the error from
  ~43 mm to ~15 mm), so this is a real geometric disagreement, not a naming
  convention.

  Independent check: in 101 of 102 photos the forearm enters from the left of
  frame — including 46 of the 47 frames the tracker called `right` — which for
  an egocentric camera means one arm was used throughout the session. **The
  tracker's handedness labels look wrong on roughly 46% of frames, and its
  3-D output is mirrored to match.** Worth raising with the professor; it is
  a known failure mode for hand trackers mounted in a non-default orientation.
  Caveat: this rests on reading the photos (arm and shoulder entering frame
  left = left arm), confirmed by eye on five frames.

- **Scale factor: median 0.88, range 0.65–1.49.** A single webcam cannot
  recover absolute hand size; MediaPipe assumes an average hand. Its
  millimetres are approximate, and the spread of that factor is the honest
  measure of how much. Fine for pose classification and shape comparison, not
  a substitute for calibrated metric ground truth.

## How fusion works

`src/cam_hand/fusion.py`. The sensors are not averaged — averaging a
measurement with a guess degrades both. Instead:

1. Scale and rotate the camera skeleton into the glove's frame, solved on the
   near-rigid palm landmarks (wrist + four knuckles).
2. Build a palm frame from the glove. A finger's direction then splits into an
   out-of-plane part (curl — the glove's) and an in-plane azimuth (spread —
   the camera's).
3. Rotate each glove finger chain rigidly about its knuckle so the azimuth
   matches the camera's, leaving curl untouched. Because it is a rotation
   about the knuckle, **every bone keeps exactly the glove's length** — the
   fused hand cannot shrink, which a per-joint blend of two point clouds
   would do (same reason the exporters use a medoid rather than a mean).
4. The thumb takes its whole direction from the camera: opposition *is*
   out-of-plane rotation, and the glove cannot see it.

Below `--min-score`, or when the camera did not see the hand, the glove
skeleton passes through untouched. Fusion degrades to glove-only, never to
garbage.

**Chirality matters and is handled explicitly.** A left hand is the mirror of
a right one, so a palm normal built from the knuckles points out of the back
of one hand and out of the palm of the other. Left unhandled, the same
physical thumb opposition gets opposite signs on the two hands and a
classifier sees one gesture as two — this was a real bug, caught by the
self-test showing `pinch` failing on right hands and `relaxed` on left.
`cam_hand.features` takes `hand_side` and flips the normal for left hands.

## Layout

```
src/cam_hand/
  landmarks.py    MediaPipe HandLandmarker -> CamHand (21 img + 21 world points)
  capture.py      webcam open/read (DirectShow first — MSMF is slow on Windows)
  draw.py         skeleton overlay + HUD (cyan left, red right, as in viz3d)
  recorder.py     JSONL record/load, per-hand rate throttling, pose/take labels
  export21.py     wrist-centring, palm synthesis, professor-format blocks
  features.py     flexion (5) + spread (6) features, LOO nearest-centroid
  align.py        Umeyama/Kabsch rigid alignment (reflections excluded)
  fusion.py       DOF-split glove+camera fusion, wall-clock frame pairing
  prof_format.py  reader for the tracker's keypoint text files
scripts/
  live_view.py               live webcam skeleton
  record_poses_cam.py        guided camera-only pose session
  record_simultaneous.py     guided glove + camera session (shared clock)
  fuse_poses.py              fuse + glove/camera/fused comparison report
  selftest_sync.py           synthetic paired dataset, answer known
  compare_to_tracker.py      camera vs the professor's tracker dataset
  export_keypoints21_cam.py  21-keypoint CSVs
  export_prof_format_cam.py  tracker-format text export
  analyze_poses_cam.py       camera pose separability report
tests/                       pytest: fusion invariants, recording, formats
models/hand_landmarker.task  MediaPipe model
```

```powershell
pytest -q
```

## Recording format (`recordings\**\*.jsonl`)

One JSON object per line, flushed per write, so an interrupted take is still
usable data:

```json
{"wall_time": 1785345678.123, "ts_ms": 123456, "hand_side": "right",
 "score": 0.98, "frame_w": 640, "frame_h": 480, "pose": "fist", "take": 1,
 "img":   [[x_px, y_px, z_rel], ...21],
 "world": [[x_m, y_m, z_m], ...21]}
```

`img` is image-space pixels (`z` is MediaPipe's relative depth). `world` is
metric-ish metres, hand-centred; everything downstream re-centres on the
wrist. Frames are never mirrored — the preview window is, the data is not.

## Known limits

- One webcam gives **approximate** millimetres (see the scale spread above).
- Occluded fingers are estimated, not measured — the glove's advantage.
- Handedness is unreliable in the egocentric views of the reference dataset,
  for both devices. Anything that mixes hands should use the chirality-aware
  features, and any conclusion that depends on left/right should be checked
  with the mirrored-refit test in `compare_to_tracker.py`.
- `selftest_sync.py` numbers are a plumbing and logic check, not evidence
  about real hardware.
