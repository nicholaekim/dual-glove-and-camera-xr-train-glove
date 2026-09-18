# Dual-sensor hand tracking: StretchSense glove + webcam
by Nicholas Kim. All rights reserved.

Both halves of the hand-tracking work in one place:

  `src/xr_hand`   the StretchSense glove pipeline (OSC receive -> validate ->
                  parse -> forward kinematics -> record/export). Started as
                  the July feasibility project; that repo is merged in under
                  `archive/summer-xr-trainer/` with its full history (July
                  dataset, technical PDF) and is frozen there.
  `src/cam_hand`  the webcam pipeline (MediaPipe 21 landmarks), plus fusion,
                  alignment, features and the comparison tooling.
  `src/leap_hand` the Ultraleap Stereo IR 170 pipeline (LeapC -> the same 26
                  OpenXR joints the glove uses). Metric 3D hand position and
                  orientation, which neither of the other two measures.
                  Plan and phases: `docs/ultraleap_ir170_plan.md`.

Glove-side scripts live in `scripts/glove/`, Ultraleap scripts in
`scripts/leap/`, webcam scripts in `scripts/`. Everything runs from this
folder with this venv.

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
| Metric 3D position, wrist angle, bone lengths | **Ultraleap** | a webcam cannot recover absolute scale; the IR camera measures millimetres |

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

That installs all three packages (`xr_hand`, `cam_hand` and `leap_hand`)
editable from `src/`. The Ultraleap bindings are **not** installed by that
line and cannot be — see "Daily use — Ultraleap" below.

Then fetch the hand-landmark model (a 7.5 MB MediaPipe binary, not kept in
this repo) into `models\`. Note `--ssl-no-revoke`: the same certificate
workaround the earlier vision experiment needed on this network.

```bash
curl.exe --ssl-no-revoke -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Check the install with `pytest -q` (expect 77 passed — camera, glove and
Ultraleap suites; a handful of the Ultraleap tests check the installed
bindings and skip when they are absent).

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

## Daily use — glove

Needs XR Trainer streaming to `127.0.0.1:9002`, glove connected and calibrated.

```powershell
python scripts/glove/run_osc.py --dump --no-viz     # confirm the stream
python scripts/glove/run_osc.py                     # live 3D glove viewer
python scripts/glove/record_poses.py --poses open_palm,fist,index_point,thumbs_up,peace,three,pinch --takes 3 --duration 5 --prep 8 --out-dir recordings/poses_v2
python scripts/glove/analyze_poses.py recordings/poses_v2   # separability report
python scripts/glove/playback.py <file.jsonl>               # scrub a recording
```

`recordings/poses/` holds the July dataset (copied from the archived project);
record new sessions into `recordings/poses_v2` etc. so the two never mix. The
July pinch/three failures trace to take 1 not being held — hence `--prep 8`.

**Glove vs camera on identical features** (pose-level, no simultaneous
capture needed):

```powershell
python scripts/compare_sensors.py --write                          # July glove data
python scripts/compare_sensors.py --glove recordings/poses_v2 --write   # re-recorded
```

## Daily use — camera

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
streams share a wall clock. `--camera leap` swaps the webcam for the
Ultraleap Stereo IR 170 and writes to `recordings\sync\leap\` instead — that
is the Path A recorder, and it has its own section under *Daily use —
Ultraleap* below.

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

## Daily use — Ultraleap (Stereo IR 170)

**See what the camera sees.** Every real-camera script opens a live window
(`scripts/leap/camera_view.py`): the IR image mirrored like a mirror, the
fitted skeleton, tracked / NO HAND / wrong hand, palm height against the
target band, palm facing, and the script's current instruction as a caption.
A green border means record-ready. It is a second, read-only client of the
tracking service, so it also runs on its own next to anything:

```powershell
python scripts\leap\camera_view.py --hand left
```

`--no-view` turns it off; the mocks never open it.

Full plan, phases and acceptance: `docs/ultraleap_ir170_plan.md`. This section
is the operating procedure.

**Phase 0 — once per machine.** The `leap` Python bindings are not on PyPI:
they wrap `leapc_cffi`, which has to be compiled against the LeapSDK on this
machine, so `pip install` cannot do it and the `leap` extra in
`pyproject.toml` is deliberately empty.

1. Download **Ultraleap Hyperion 6.2.0**, Operating System = **Windows**,
   from https://www.ultraleap.com/downloads/sir170/ and install it. It brings
   the tracking service, the Control Panel/visualiser and the LeapSDK at
   `C:\Program Files\Ultraleap\LeapSDK`.
2. Plug the camera into a **direct USB port** — not a hub. It needs 0.5 A and
   browns out on unpowered hubs.
3. Open the **Ultraleap Control Panel**, confirm the device is listed, the
   visualiser shows hands, and the tracking mode is **Desktop**. If the panel
   does not list the SIR170, uninstall and use Gemini 5.x from the same page
   (plan section 1); the code is identical either way.
4. Build and install the bindings, then verify:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\leap\setup_bindings.ps1
   python scripts\leap\check_setup.py
   ```
   `setup_bindings.ps1` clones the bindings into the Ultraleap archive folder
   (`..\xr trainer\reference\ultraleap\`, created if missing), compiles
   `leapc_cffi` against the SDK, installs both packages into `.venv` and ends
   by running the checker. It reuses an archived wheel built for the running
   interpreter when there is one, so a repeat setup takes seconds instead of
   a compile; `-Fresh` forces a rebuild. `check_setup.py` prints a line for
   the SDK folder, `LeapC.h`, `LeapC.dll`/`.lib`, the tracking service,
   `import leap`, then **device streaming** and **hand seen** over a 6 s
   window, with the fix on every FAIL. `hand seen` is a **WARN**, not a FAIL,
   when the device streams but nobody is holding a hand over it — an empty
   room is not a broken machine, so that still exits 0; anything else failing
   exits 2. Set `LEAPSDK_INSTALL_LOCATION` first if the SDK is not at the
   default path.
5. Archive the installer and the `LeapSDK` folder into
   `..\xr trainer\reference\ultraleap\` the day they are downloaded —
   Ultraleap releases have disappeared before. The bindings clone already
   lives there and `leap` is installed **editable from it**, so that folder
   is not a copy of the working install, it *is* the working install: do not
   delete it.

   Verified on this machine: Hyperion `6.2.0+2025.07.11`, and `leapc_cffi`
   compiles cleanly on Python 3.14 (`leapc_cffi-0.0.1-cp314-cp314-win_amd64`),
   so the 3.11 fallback venv in plan section 4 is not needed.

**Daily.**
```powershell
python scripts\leap\live_view.py                 # 3D skeleton + tracking HUD
python scripts\leap\record_poses.py              # 6 poses x 3 takes x 5 s
python scripts\leap\record_poses.py --raw        # also write LeapC .lmt files
python scripts\leap\record_frame.py 128166       # one professor reference frame
python scripts\leap\stats.py recordings\leap\poses --write
```
`live_view.py` is the first thing to run in a session: with an open palm over
the module it shows whether finger order, chirality and fingertips are right.
`record_poses.py` is the glove's guided beep protocol, same pose list and
filenames, writing to `recordings\leap\poses\`. `record_frame.py` is the
camera's version of July's `record_frame.ps1`: it shows the professor's
reference image, records a few seconds, and writes the medoid frame in his
keypoint format to `recordings\leap\prof_frames\`.

**Gate experiment (Phase 2).** The one experiment that decides the
architecture: does the IR tracker see a hand inside the black StretchSense
glove? One command runs the whole of plan section 2 —

```powershell
python scripts\leap\gate.py
```

— four conditions in order (`bare`, `glove`, `glove_liner`, `glove_tape`),
each running the same **timed pose schedule** (`--schedule`, default
`open_palm:5,fist:5,pinch:5,spread:5`). Per condition it counts you in with
beeps, calls each pose as its window starts, records, takes IR stills spread
across the run, then prints what to change on your hand and waits for Enter.
The same one-line HUD as the Path A recorder runs throughout — pose being
called for, hand tracked, palm height against `--band`, viewing angle — and
out of band is flagged and counted, never blocked. Add `--raw` for LeapC
`.lmt` files, or narrow it with `--conditions bare,glove`. It writes:

```
recordings\leap\gate\<condition>\   the JSONL take, the IR stills (PNG + JSON
                                    sidecar each), and .lmt with --raw
results\leap_gate\REPORT.txt        one table row per condition and hand, a
                                    per-pose block, the paired per-pose
                                    verdict, then "Path A: yes/no because ..."
```

The condition verdict applies the plan's thresholds — detection >= 80 %, at
most one re-acquisition per 10 s, jitter within 2x the bare hand of the same
side. Path A means simultaneous glove + camera capture; Path B means
sequential.

**Why the schedule.** On 2026-09-17 the operator chose poses and heights
freely, and every number came out uninterpretable: the bare baseline scored
62 % because the hand turned edge-on for five seconds, one gloved run sat
99 mm above the lens (below the device's range), the runs labelled "35cm" and
"50cm" were actually at 198-240 mm and 267 mm, and four gloved runs lost
tracking while the hand was a fist — with **no bare-hand fist anywhere to
compare against**. Now each frame carries the plan it was recorded under and
its own offset into it, so the report measures each pose separately and pairs
each gloved pose against the bare hand doing **the same pose on the same
hand**:

```
Paired verdict per pose (glove vs bare, same hand)
  glove   left  open_palm  PASS     detection 96.1% (bare 98.4%), longest loss 0.08 s
  glove   left  fist       FAIL     detection 41.0% is below 80% and 57.0 points below bare
  glove   left  pinch      NO BARE  ... there is no bare pinch for the left hand to pair
```

A pose passes when glove detection is >= 80 % **or** within 10 points of bare,
**and** its longest loss is <= 1 s **or** no longer than bare's — absolute or
relative on both clauses, because a pose the camera cannot see bare is not
evidence about the glove. A pose with no bare partner is reported as missing
and left unjudged. A condition named `<name>_<N>cm` sets its own band to
N +/- 5 cm and is paired against `bare_<N>cm` where that run exists, so a
distance sweep measures the glove rather than the height.

**Result (2026-09-16): Path A, provisionally.** The IR camera tracks the hand
*inside* the black StretchSense glove in **100 %** of frames, with **0**
re-acquisitions and **0.20 mm** fingertip jitter against the bare hand's
0.54 mm. In the 850 nm stills the glove renders almost white with its sensor
grid visible as dots, so the fabric reflects near IR strongly even though it
is black to the eye. **Do not read the better-than-bare jitter as a property
of the fabric**: black textiles vary widely in near-IR reflectance and one
session cannot separate that from run-to-run variation — which is what the
extra runs below are for. The bare left hand scored 89.2 %. Every number is
from that one session and lives in `results\leap_gate\REPORT.txt`; the
detection rates are the corrected ones, recomputed from the same recordings
— see the next paragraph. Plan section 9 has the full entry.

**Recompute a report without the camera.**

```powershell
python scripts\leap\gate.py --recompute
```

Rebuilds `results\leap_gate\REPORT.txt` from the recordings and IR sidecars
already under `--out-dir`, discovering the conditions from the folder names
and measuring the most recent take in each (older takes are named in the
report, never averaged in). Use it whenever a measurement changes, so an
earlier session gets the corrected numbers without anyone re-running the
protocol. It has been used once already: the detection denominator is now the
LeapC tracking framerate for a file that kept every frame, because the
10th-percentile gap of a full-rate file measures the timestamp jitter rather
than the cadence (it read 101 Hz on a 90 Hz file). That moved bare left from
79.2 % to 89.2 % and the glove from 98.6 % to 100 %, and left the verdict
where it was. `leap_hand.stats.choose_rate` is the rule; the report's
footnote names the denominator every row used.

Takes recorded before schedules existed have no pose boundaries in them, so
`--recompute` reads them exactly as it always did — the per-condition table,
unchanged — and says why there is no per-pose block rather than attributing
frames to a pose by where they sit in the file. Re-run the gate to get the
per-pose comparison.

**Extra gate runs the reviewer asked for** (plan section 9). A centred
right-hand glove run and one with both hands up at once, then the gloved hand
held at 20, 35 and 50 cm above the module, working through **fist, pinch and
open/spread** in each distance run — then the whole set repeated on a second
day, so the result is not one session's lighting:

```powershell
python scripts\leap\gate.py --conditions bare,glove_right,glove_both,glove_20cm,glove_35cm,glove_50cm
```

Condition names are free-form: `bare` is the reference, everything else is a
gloved condition judged against the bare hand **of the same side**, so these
need no code change — show the bare hand on both sides in the `bare` run and
each side's glove row is compared against its own baseline. Repeat the next
day into a separate `--out-dir` (add `glove_day2` if you want the second
day's plain-glove run in the same table), then `--recompute` either folder
for its table.

**The IR stills are the evidence.** A `.lmt` and our JSONL both hold *solved
skeletons*, so when the tracker sees nothing they are empty and prove
nothing. The stills are the raw 850 nm images of the glove, and each PNG has
a sidecar naming the hands the tracker reported at that instant — a photo of
a clearly visible glove next to `"no hand tracked"` is itself the finding.
Take them on their own with:

```powershell
python scripts\leap\ir_snapshot.py --label glove --count 10 --interval 0.5
```

Measured on this unit: 384 x 384, 8-bit, one plane per eye.

**Convention checker.** `python scripts\leap\convention_check.py` holds a
real hand in front of the device and verifies every LeapC convention the
26-joint mapping is built on (bone `rotation * (0,0,-1)` along
`prev_joint -> next_joint`, the zero-length thumb metacarpal, the palm basis,
fingertips farthest from the wrist). Exit 0 all good, 2 no hand appeared,
3 a convention failed — in which case do not collect data until
`to_openxr.py` is corrected. It came back all zeros on this unit; see plan
section 9.

### Path A: recording glove and camera together

The gate said Path A, which means one hand, one instant, both sensors — the
glove supplying flexion, the camera supplying spread, thumb opposition, palm
pose and wrist. Two commands, in order. Wear the glove on **one** hand, keep
the other out of the module's field, and have XR Trainer streaming to
`127.0.0.1:9002`.

```powershell
python scripts\record_simultaneous.py --camera leap --hand left
python scripts\fuse_poses.py recordings\sync --write
```

`--hand` is required with `--camera leap`, and it is the whole protocol.
Measured on 2026-09-17, left gloved hand: `open_palm` tracked 3/3 takes on one
continuous hand id, and **every other pose failed** — fist 34 %/0 %/66 %,
index_point 0/8 %/tracked-but-labelled-RIGHT, thumbs_up 0/0/0, pinch 0 as
left, peace 1/3. The tracker will follow an open hand into a pose, but it
cannot acquire a gloved hand that is already closed, and when it re-acquires
from a closed pose it sometimes returns a **mirrored skeleton labelled as the
other hand** — which fuses against the other glove and produces a plausible,
wrong result. The idle other hand, 20 cm off to the side, was picked up too.

**What the operator sees.** Each take is three phases with one live line,
rewritten in place about four times a second:

```
ACQUIRE    --  left YES      24.3 cm OK       view  12 deg  glove 60.2/s
SETTLE    0.9s left YES      24.1 cm OK       view  14 deg  glove 60.1/s
REC       3.4s left YES      23.8 cm OK       view  11 deg  glove 59.8/s
```

* **ACQUIRE** prints `hold the LEFT hand OPEN PALM over the camera` plus a
  per-pose hint where the pose has a known trap (thumbs_up: tilt the whole
  forearm 30-45 degrees so the camera still sees some palm; pinch: palm
  facing the lens; fist: close slowly). Nothing is recorded and the pose is
  never called until the expected hand is tracked for half a second, inside
  the height band (`--band`, default 18-28 cm — the takes that worked sat at
  16-21 cm, the ones that failed at 13-15 cm), with the palm within 40 degrees
  of facing the lens and roughly over the module. The line says which of those
  is missing (`need: TOO LOW, turn palm to lens`), and `WRONG HAND` when the
  only thing in view is the other chirality.
* **SETTLE** beeps, prints `NOW: FIST`, and gives `--settle` seconds (default
  1.5) for the hand to change shape while the tracker follows it.
* **REC** records `--duration` seconds — but only if the hand id pinned at
  acquire is still on the hand. If it changed, or the hand was lost, the
  attempt says so, **both files are set aside** (see below), and the take is
  retried from ACQUIRE up to `--retries` times (default 3). A take is complete
  when the expected hand, on that one id, covers >= 90 % of it — **and** the
  hand was in the pose that was asked for.

**The window names the pose, and a take whose hand is not in it is refused.**
On 2026-09-17 the window's caption was phase + extras + seconds, and the
extras are empty during SETTLE and REC, so the window the operator was
actually watching said `SETTLE   1s` and never once named the pose — `NOW:
FIST` went to a terminal nobody was looking at. Nine of that session's 36
takes hold the wrong pose, **with both sensors agreeing on the wrong one**
(three fists that are open palms, three index points that are fists, two
thumbs ups that are fists, one peace that is a thumbs up). So the caption is
now a tested function and says the pose in every phase:

```
ACQUIRE   OPEN PALM first   next: FIST (take 2/3)
SETTLE    NOW: FIST   1s
REC       HOLD: FIST   REC 4s
refused   WRONG POSE: saw OPEN HAND, want FIST
```

and after the take `leap_hand.pose_check` compares each finger's curl
(tip-to-wrist over palm length — the same number `fuse_poses.py` reports)
against the shape the pose is supposed to be. **A finger is only wrong when
BOTH sensors are decisive and BOTH contradict the pose**; one sensor
disagreeing is a warning and never a failure. That rule is not caution for
its own sake, it is what the same session measured: the glove reports an
exact open palm for every real pinch (it cannot see thumb opposition at all),
and the camera reads the ring finger extended on the right-hand peace takes
where the glove reads it curled. So `pinch` is judged on the camera's
thumb-index gap alone, reported as `pinch_camera_gap` and **never rejected
on** — the camera may not select the pinch takes it is later going to be
scored on — and a camera looking at the hand edge-on (median viewing angle
over 65 degrees) casts no vote at all, so nothing can fail on it.
`--no-pose-check` turns the whole thing off; it is off automatically for
`--mock-glove`/`--mock-leap`, whose cartoon hands do not follow the pose
being called.

**A refused attempt is never deleted.** Both files move to
`recordings\sync\rejected\glove\` and `...\rejected\leap\` under the same name
plus `_attemptN`, with a `meta.json` saying `accepted: false` and why — the
two sensors under evaluation are the ones vetoing the take, so every
exclusion has to stay countable. `fuse_poses.py` and `check_take_labels.py`
both ignore `rejected/`. The session summary ends with how many attempts the
pose check refused, per pose.

**One still per take, as independent evidence.** At the midpoint of REC the
camera window saves the frame it is composing — IR image, fitted skeleton,
caption — to `recordings\sync\stills\<take>.jpg` (and to `rejected\stills\`
with a refused attempt), through a `snap=<path>` line in the status file it
is already driven by. It is the only record of a take that does not come from
the two sensors that judge it. `recordings\` is git-ignored and the stills
have the operator in them, so they never go anywhere else.

**The glove is recorded at full rate** with `--camera leap` (`--hz` still
throttles it explicitly, and the webcam backend still defaults to 5 Hz). At
5 Hz a five-second take is 24 glove frames, which cannot tell a steady stream
from one that stopped for three seconds; the meta files now carry
`glove_rate_hz`, `glove_max_gap_ms` and `glove_gaps_over_100ms`, and a take
whose glove went quiet for more than 250 ms says so as it finishes. Pairing
is unaffected — every glove frame is matched to the nearest camera frame, and
a mock session at full rate pairs 94 % of them within 50 ms.

Audit a folder that was recorded before any of this existed:

```powershell
python scripts\check_take_labels.py recordings\sync_coached_20260917
```

It prints every take's verdict, both sensors' median curls, the glove's
stream health and one line of English (`both sensors saw an OPEN HAND,
expected FIST`), then the exact command to re-record the poses that were
wrong — whole poses, because take numbering is per pose. On that session it
reproduces the nine wrong takes exactly, warns on the four the two sensors
disagree about, and flags no pinch.

A camera hand of the other chirality is dropped before it can reach the
recorder, so `--hand left` cannot produce a right-handed line; the wrong-hand
and second-hand-in-view counts are printed per session and stored per take.
Each take writes two files under **one name**, `recordings\sync\glove\...` and
`recordings\sync\leap\...`, both stamped with `time.time()` at the write — the
one clock the sensors share — plus `<take>.meta.json` beside the camera file
with `accepted`, the hand, pose, hand id, coverage, attempts, median height,
median viewing angle, the rejection counts, the glove's stream health, the
still's path and the whole `pose_check` verdict. The session summary ends with
the exact command to redo only the poses that failed:

```
  open_palm    3/3 complete (3 attempt(s))
  thumbs_up    0/3 complete (12 attempt(s))   last failure: the tracker let go
                                              and re-acquired the hand (id 12 -> 15)

  Redo ONLY the poses that failed:
    python scripts/record_simultaneous.py --camera leap --hand left --poses thumbs_up --takes 3
```

`--hand both` is the old uncoached behaviour: both hands, no acquire phase,
the once-a-second HUD. There is no preview window either way — your hands are
over the module and an 850 nm brightness image is not something to check a
pose against. The camera is **not** throttled by `--hz` (which stays the
glove's rate): keeping every frame at 90 Hz is what guarantees each glove
frame a partner within a few milliseconds, and `--leap-hz` overrides that if
disk ever matters more than pairing.

The second matches the frames by wall clock, fuses each pair and prints the
three-row table — glove only, camera only, fused — plus how many frames found
a partner and how often the camera actually contributed. It recognises a leap
take by the `source: "leap"` key in the file itself, not by the folder, and
aligns it onto the glove **rigidly, without scale**, because Leap joints are
real millimetres and a scale factor would quietly absorb the difference
between the hand and the glove's template skeleton (plan section 3). A
MediaPipe take in `recordings\sync\cam\` keeps its Umeyama-with-scale fit, and
both kinds can sit in one session folder.

Rehearse the whole thing with no hardware at all:

```powershell
python scripts\record_simultaneous.py --camera leap --hand left --mock-glove --mock-leap --poses fist --takes 1 --duration 3
python scripts\fuse_poses.py recordings\sync
```

Single reference frames — the 21 poses the glove could not do — go through
`scripts\leap\record_frame.py 128166`, which shows the professor's image,
records `--seconds` (default 3), picks the medoid frame and writes
`<frame>_keypoints.txt` in his format into `recordings\leap\prof_frames\`.

**No camera to hand?** Every script here takes `--mock`, which drives the same
code with synthetic hands at 90 Hz:
```powershell
python scripts\leap\check_setup.py --mock
python scripts\leap\live_view.py --mock --no-window --duration 5
python scripts\leap\record_poses.py --mock --takes 1 --duration 2 --prep 1 --poses open_palm,fist
python scripts\leap\ir_snapshot.py --mock --label demo --count 2
python scripts\leap\gate.py --mock --conditions bare,glove --seconds 3 --prep 0
python scripts\leap\record_frame.py 128166 --mock --prep 0 --seconds 1
python scripts\record_simultaneous.py --camera leap --hand left --mock-glove --mock-leap --poses fist --takes 1 --duration 3
python scripts\leap\stats.py --mock
```
`--mock` images are obviously synthetic gradients and their sidecars say so;
`convention_check.py` has no mock, because a convention can only be checked
against a real device.

**Exports — the existing tools, unchanged.** Leap recordings are JSONL with
exactly the glove's keys plus camera-only extras, and `FrameRecorder.load`
ignores keys it does not know, so:
```powershell
python scripts\glove\playback.py recordings\leap\poses\<file>.jsonl
python scripts\glove\export_keypoints21.py recordings\leap\poses
python scripts\glove\export_prof_format.py recordings\leap\poses
```
Two consequences of Leap frames carrying the hand's **real position** in
camera space (a wrist 25 cm above the module), where a glove frame carries
nothing outside the wrist:

  * `playback.py`'s view box is fixed at ±0.22 m around the origin, so the
    hand can sit outside the view. `scripts\leap\live_view.py` re-anchors the
    wrist and does not have this problem.
  * `export_prof_format.py` writes absolute camera millimetres, so its
    `Wrist:` line is a real position instead of the glove's `(0, 0, 0)`. The
    landmark block is unchanged in layout, and `compare_to_tracker.py` aligns
    rigidly before scoring, so this only matters if a file is read by eye.

`export_keypoints21.py` is wrist-centred either way and is unaffected.

**Units and frame.** Positions are metres, the same unit as glove
`HandFrame`s (LeapC's millimetres are converted once, in
`leap_hand.to_openxr.from_leap_mm`). The frame is LeapC desktop-mode camera
space: right-handed, origin at the module, +x along the baseline, +y up, +z
toward the user. `hand.confidence` is never used anywhere — LeapC documents
it as a constant 1.0; recordings are gated on presence and `visible_time`
instead. The stream refuses to run unless the service confirms Desktop mode,
because a recording made in the wrong mode is in the wrong frame.

**Two clocks in every recording**, and they are not interchangeable:

  * `wall_time` — `time.time()` as the line is written, the same stamp the
    glove and camera recorders use. This is the **only clock the three
    sensors share**, and it is what `fuse_poses.py` pairs takes on.
  * `timestamp` (and `timestamp_us`) — `event.timestamp`, the **LeapC
    clock**, whose epoch is arbitrary. Right for intervals inside one
    recording, since it carries none of the jitter our writer adds — that is
    why `stats.py` measures cadence from it — and meaningless compared
    against wall time or another sensor. The glove's `timestamp` is likewise
    its own tick counter, so this matches the existing convention.

`frame_age_us` is `leap.get_now() - event.timestamp` sampled in the tracking
callback: how old the data already was when we received it. Frame age, not
end-to-end latency.

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

## Teaching a detector to see the gloved hand

Since MediaPipe cannot see the glove (above), the fix is a detector trained on
gloved hands. The obstacle is labels: that needs gloved images with 21
keypoints marked, and marking them by hand is thousands of clicks.

The way around it is that **repainting a hand does not move its landmarks**.
The Ultralytics hand-keypoints dataset ships 26,768 bare-hand images with
ground-truth 21-keypoint labels; painting each hand to look gloved
(`cam_hand.synth_glove`, using the labels themselves to build the hand mask)
produces gloved training images whose existing labels are still exactly
correct.

Checked before relying on it: MediaPipe detects the bare source images at
**100%** and only **10%** after repainting — the same collapse the real glove
causes. The synthetic glove reproduces the actual failure rather than just
looking dark.

```powershell
python scripts/build_glove_dataset.py --n 2500 --extra-dir reference/frames --extra-repeat 4
```

Writes a standard YOLO pose dataset to `datasets/` (gitignored). `--preview`
adds a bare-vs-gloved contact sheet.

`--extra-dir` folds in extra bare-hand photos that have no labels of their own
— the reference tracker frames are the useful case, being the only images from
the actual lab scene. They are **pseudo-labelled** with MediaPipe (which
detects them at 100%), so those labels are one model's output rather than
ground truth and carry its errors; `--extra-repeat` repeats each with fresh
fabric jitter so a small in-domain set is not drowned by the generic one. Note
the professor's own `.txt` files cannot serve as labels here: they are 3-D
millimetres in his tracker's frame, not image coordinates.

Then train and predict. Both scripts import ultralytics and nothing from this
project, so run them with the **old project's** interpreter, which already has
ultralytics and `yolo11n-pose.pt`:

```powershell
& "..\xr trainer\summer-xr-trainer\.venv\Scripts\python.exe" scripts/train_glove_model.py --smoke
& "..\xr trainer\summer-xr-trainer\.venv\Scripts\python.exe" scripts/train_glove_model.py --epochs 30 --imgsz 512
& "..\xr trainer\summer-xr-trainer\.venv\Scripts\python.exe" scripts/predict_glove.py --source cam
```

That torch install is **CPU-only**, so a real run is hours, not minutes —
`--smoke` (2 epochs at 320 px) proves the loop completes first. Expect pose
mAP near zero early on: the pretrained head covers 17 body keypoints and is
reinitialised for 21 hand keypoints, so keypoint accuracy starts from scratch
while box detection transfers immediately.

`predict_glove.py --source cam` is the test that counts — put the glove on and
see whether the skeleton tracks, against MediaPipe's measured 0%.

**The sim-to-real gap, measured the hard way.** The first model validated at
pose mAP50 0.728 on held-out *synthetic* gloves — and scored **0/250 frames on
the real glove** (`predict_glove.py --source cam`). Real captured frames
(`scripts/capture_frames.py`) showed why: the v1 repaint drew a closed dark
mitten with a dot grid, while the real glove is **open-tipped with bare skin
at the fingertips**, keeps fingers visibly separated, and carries a round
sensor puck and a wrist strap. The model learned the wrong object.

The v2 repaint is drawn from those real frames (tips bare, fingers separate,
puck, strap, ribbed knit, near-neutral colour). Lesson worth keeping: fooling
MediaPipe is a weak validation of a synthetic glove — match the *actual
artifact*, and test on real frames as early as possible.

Real-frame test sets, captured with `capture_frames.py`: `captures/gloved` for
detection rate, `captures/bare` as the control (MediaPipe manages only 50%
even on bare hands in this room's backlight — the scene itself is hard).

## How fusion works

`src/cam_hand/fusion.py`. The sensors are not averaged — averaging a
measurement with a guess degrades both. Instead:

1. Scale and rotate the camera skeleton into the glove's frame, solved on the
   near-rigid palm landmarks (wrist + four knuckles).
2. Build a palm frame from the glove. A finger's direction then splits into an
   out-of-plane part (curl — the glove's) and an in-plane azimuth (spread —
   the camera's). The azimuth is read off the **proximal bone** (knuckle ->
   PIP) on both sides, not knuckle -> tip: abduction happens at the knuckle,
   so once the PIP and DIP are bent the tip swings far off its own bone and a
   few degrees of flexion error would arrive as tens of degrees of fake
   abduction.
3. Rotate each glove finger chain about its knuckle, **about the palm
   normal**, until its azimuth matches the camera's. A rotation about the
   normal changes only the azimuth, so the glove's curl survives exactly, and
   because it is a rotation about the knuckle **every bone keeps exactly the
   glove's length** — the fused hand cannot shrink, which a per-joint blend of
   two point clouds would do (same reason the exporters use a medoid rather
   than a mean).
4. The thumb takes its whole direction from the camera: opposition *is*
   out-of-plane rotation, and the glove cannot see it.

Below `--min-score`, or when the camera did not see the hand, the glove
skeleton passes through untouched. Fusion degrades to glove-only, never to
garbage.

### Which sensor is right on THIS frame

Steps 1-4 say which sensor *owns* a degree of freedom. They do not say whether
the camera actually measured it on this frame, and the first real simultaneous
session showed that the difference decides the result:

| pose | glove | camera | who is right |
|------|-------|--------|--------------|
| `pinch` | index curl 1.97, thumb-index gap 0.99 — **numerically its own open palm** | index curl 1.31, gap 0.46 | camera |
| `thumbs_up` | four fingers curled, 0.66 | four fingers nearly straight, 1.68, `grab_strength` 0.04 | glove |

Taking the camera whenever a camera frame existed imported the second case
along with the first, and the fused classifier scored *below* glove-only. Each
camera-owned DOF is now gated per frame, on evidence that is not the tracker's
opinion of itself:

- **frame level** — the hand has been visible for `min_visible_time_us`
  (300 ms), its `hand_id` has not changed within `hand_id_settle_s` (0.25 s),
  and the palm is inside the module's central field (lateral offset less than
  height, i.e. within `field_half_angle_deg` = 45 degrees of vertical).
- **spread, per finger** — only when the *glove* says that finger is not
  strongly curled (curl above `curl_gate` = 1.2, the midpoint between the
  glove's fist and open-palm values; a curled finger has no abduction left to
  see and its proximal bone points end-on at the camera) **and** the palm is
  turned toward the module (`view_gate_deg` = 50, measured between the
  camera's palm normal and the ray from the palm to the module, which is the
  origin of leap space).
- **thumb** — only when the viewing-angle gate passes **and** the camera
  agrees with the glove about index..little: median absolute curl
  disagreement below `curl_agree_tol` = 0.35. A camera that has the four
  fingers wrong has the hand's orientation wrong, and orientation error moves
  the thumb most of all. `pinch` measures 0.24-0.28 and passes; `thumbs_up`
  measures 0.88 and does not — and was edge-on at 70-78 degrees, so it fails
  twice over.

`grab_strength`, `pinch_strength` and `confidence` are deliberately **not**
used as weights: the first two are outputs of the same model that produced the
joints, so they cannot corroborate it, and LeapC reports `confidence` as a
constant 1.0.

Every threshold is a named field of `fusion.GateParams`, overridable on the
command line (`--curl-gate`, `--view-gate-deg`, `--curl-agree-tol`) and
printed in every report. They are **empirical starting points read off one
session**, not calibrated constants.

No frame is ever dropped: a frame that fails every gate is the glove
skeleton, unchanged. `fuse_skeletons` returns, per frame, `dof_source` (which
DOFs the camera supplied) and `rejected` (why the glove kept the rest), which
is what the report counts. A MediaPipe frame carries none of these capture
facts — no absolute palm, no hand id, no visibility clock — so it passes
`cam_meta=None` and is fused ungated, exactly as before; gating a sensor on
evidence it does not produce would mean rejecting all of it.

### When the GLOVE is wrong: the rail-disagreement override

The split above hands the glove every finger curl, and on
`recordings/sync_day1` that is wrong in one specific, reproducible way.

Whenever a finger is straight the glove does not report a measurement, it
reports a **constant** — index 1.97 (left) / 1.98 (right), middle 2.07/2.08,
ring 1.97, pinky 1.71, thumb 1.43, identical to two decimals on every
open-palm frame of all 59 takes. That constant is the finger's **rail**: the
top of the stretch sensor's range, where the fabric has stopped stretching and
the number has stopped meaning anything.

In all ten pinch takes the glove index sits exactly on its rail while the
camera watches the index fold down to the thumb (camera index curl 1.16-1.39,
against 1.72-1.81 for a genuinely open palm; thumb-index tip gap 0.11-0.28
palm lengths). The fused pinch therefore had a perfectly straight index — the
one joint the gesture is named after.

This is **not** a dead zone. An isolated slow index bend leaves the rail as
soon as the camera sees any flexion, so the glove does measure that finger; it
fails only at the very top of its range. So the override is narrow, and all
three conditions must hold together, per finger, per frame:

- **on the rail** — the glove's curl is within `tol` (0.005) of the rail
  *learned* for that hand and finger. The value is bit-exact, so this is a
  float-equality test, not a band.
- **camera trusted** — the same frame gates the spread and thumb already use:
  visible time, no recent `hand_id` change, central field, and viewing angle
  inside `view_gate_deg`.
- **camera flexed** — the camera's curl for that finger is below its open
  reference minus `margin` (0.25), i.e. 1.50 for the index.

...and then hysteresis, because one frame is not evidence: the override arms
only after `enter_frames` (10, about 170 ms at the glove's 60 Hz) consecutive
qualifying frames and disarms after `exit_frames` (5) non-qualifying ones. A
fresh tracker per take, since "consecutive" across a take boundary is a
fiction.

**Why the rail is learned and the camera's open reference is not.** The rail
is a *sensor artefact*: there is no physical reason index saturates at 1.97
and pinky at 1.71, or that the two hands differ in the third decimal. Those
are facts about this glove and the only way to know them is to look, so
`learn_rails` finds them as the most frequent curl per hand and finger — a
saturating sensor puts a huge bit-exact spike where a moving one spreads out —
and refuses to believe one unless it also sits at the top of the observed
range, so a session that never shows a finger straight teaches no rail and the
override never arms.

The camera's open reference is a *geometric fact*, so it is a constant
(`cam_open_curl`). Curl here is fingertip-to-wrist over palm length — a
dimensionless ratio — so a straight finger reads about the same on any hand:
on sync_day1's open-palm frames the two hands differ by 0.03 (index) to 0.05
(pinky), an order of magnitude under the margin. It must **not** be learned
from the frames being fused, and that is the point: without pose labels the
only label-free way to call a frame "open" is to ask the glove, and the glove
saying "open" is exactly the claim under suspicion. A reference learned that
way would let a session of nothing but pinches teach the detector that a
pinched index is what open looks like. A constant cannot be poisoned by the
data it is judging.

**What it does.** The finger is rebuilt from the glove's knuckle with the
camera's three bone *directions* and the glove's three bone *lengths*
(`transfer_finger_flexion`) — the spread transfer's idea one step further,
where a single rotation about the knuckle becomes each bone in turn adopting
the camera's direction. Bone lengths are preserved exactly, as everywhere else
in this module.

One consequence shows up in every report and is worth stating plainly: **the
fused curl does not land on the camera's number.** It lands about 11% above
it, because the curl metric is a *length* ratio and the glove's template index
is about 11% longer per palm length than the operator's (rail 1.97 against the
camera's 1.75 open). Over the ten pinch takes the fused index reads 1.29-1.56
against the camera's 1.16-1.39. The joint *angles* are the camera's exactly —
expressed as a fraction of each sensor's own open value the two agree to
within 0.01 — and only the bones they hang on are the glove's. Making the
number match outright would mean rescaling the template, which is the one
thing this module never does.

Defaults: **on, index only**. `--no-rail-override` turns it off,
`--rail-fingers index,middle,...` widens it. Every threshold is a named field
of `fusion.RailOverrideParams` and is printed in the report, along with the
rails actually learned.

**Measured on `recordings/sync_day1`** (59 takes, `--camera leap`): the
override is active on 77.9% of paired pinch frames and 0.00% of fist,
index_point, peace and thumbs_up frames. The only non-pinch activation is 23
contiguous frames (0.77%) near the end of one open_palm take, where the camera
saw the index at 0.90 — fist-like — and the glove had itself left the rail for
part of it, i.e. a real movement at the end of the take rather than a false
positive.

### What the report says

`scripts/fuse_poses.py` leads with a **per-DOF table**: for each pose, hand
and take, the glove / camera / fused value of the five curls, the four
adjacent proximal-bone spreads and the thumb-index gap, with a `from` column
saying how much of each row the camera actually supplied. Under it are the
camera-use rate per gated DOF, every rejection reason with its count, and the
thresholds in force.

The leave-one-out classifier is below that, as a **secondary** metric, and it
is now leave-one-**TAKE**-out: holding out one *sample* left the other hand of
the same five seconds in the training set, which mostly measured whether a
person's two hands look alike. With only one take per pose the held-out pose
has no centroid at all and the number is meaningless; the report says so
instead of quoting it.

**Chirality matters and is handled explicitly.** A left hand is the mirror of
a right one, so a palm normal built from the knuckles points out of the back
of one hand and out of the palm of the other. Left unhandled, the same
physical thumb opposition gets opposite signs on the two hands and a
classifier sees one gesture as two — this was a real bug, caught by the
self-test showing `pinch` failing on right hands and `relaxed` on left.
`cam_hand.features` takes `hand_side` and flips the normal for left hands.

## Layout

```
src/xr_hand/            glove pipeline (joints, parser, validator, receiver,
                        recorder, kinematics, keypoints21, mock, viewers)
scripts/glove/          glove scripts (record_poses, run_osc, analyze_poses,
                        playback, exports, mock utilities)
src/cam_hand/
  landmarks.py    MediaPipe HandLandmarker -> CamHand (21 img + 21 world points)
  capture.py      webcam open/read (DirectShow first — MSMF is slow on Windows)
  draw.py         skeleton overlay + HUD (cyan left, red right, as in viz3d)
  recorder.py     JSONL record/load, per-hand rate throttling, pose/take labels
  export21.py     wrist-centring, palm synthesis, professor-format blocks
  enhance.py      optional gamma / local-contrast boost before detection
  synth_glove.py  repaint a bare hand as gloved, keeping its labels
  features.py     flexion (5) + spread (6) features, LOO nearest-centroid
  align.py        Umeyama/Kabsch rigid alignment (reflections excluded)
  fusion.py       DOF-split glove+camera fusion, wall-clock frame pairing
  prof_format.py  reader for the tracker's keypoint text files
scripts/
  live_view.py               live webcam skeleton
  tune_detection.py          sweep settings to detect a hand being missed
  capture_frames.py          save raw webcam frames (real-glove test sets)
  build_glove_dataset.py     labelled bare hands -> gloved training set
  train_glove_model.py       fine-tune YOLO pose on it (old venv)
  predict_glove.py           run the trained model on the real glove
  record_poses_cam.py        guided camera-only pose session
  record_simultaneous.py     guided glove + camera session (shared clock);
                             coached one-hand protocol with --camera leap
  fuse_poses.py              fuse + glove/camera/fused comparison report
  selftest_sync.py           synthetic paired dataset, answer known
  compare_sensors.py         glove vs camera, same features, same classifier
  compare_to_tracker.py      camera vs the professor's tracker dataset
  export_keypoints21_cam.py  21-keypoint CSVs
  export_prof_format_cam.py  tracker-format text export
  analyze_poses_cam.py       camera pose separability report
tests/                       pytest: fusion invariants, recording, formats
models/hand_landmarker.task  MediaPipe model
docs/ultraleap_ir170_plan.md Ultraleap Stereo IR 170 integration plan (next phase)
archive/summer-xr-trainer/   July glove-only repo, merged with history: the
                             42-take July dataset, REPORT.txt, technical PDF,
                             record_frame.ps1 (frozen, see archive/README.md)
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

## Repository history

- July 2026: glove-only pipeline at https://github.com/nicholaekim/summer-xr-trainer
  (now read-only; merged here on 2026-09-16 under `archive/summer-xr-trainer/`
  with all of its commits).
- August 2026: this repo, https://github.com/nicholaekim/dual-glove-and-camera-xr-train-glove,
  adds the webcam pipeline, fusion and the tracker comparison; the glove code
  was consolidated into `src/xr_hand` and `scripts/glove/`.
- September 2026: Ultraleap Stereo IR 170 planned as the second sensor, see
  `docs/ultraleap_ir170_plan.md`.

## Known limits

- **MediaPipe cannot see a hand wearing the black StretchSense glove.**
  Measured, not estimated (`scripts/tune_detection.py`, 40 frames per
  configuration, same lighting and position for both conditions):

  | condition | detection rate |
  |---|---|
  | gloved hand, all 8 configurations | **0 / 320 frames (0%)** |
  | bare hand, threshold 0.50 → 0.05 | **320 / 320 frames (100%)** |

  Lowering the confidence floor as far as 0.05 recovers nothing, so the model
  is not finding the hand weakly — it is not finding it at all. It was trained
  on bare skin and the glove does not read as a hand. This is the sharpest
  obstacle to fusion, because the camera has to see the hand *while* the glove
  is on it. Full table in `results/detection_tuning.csv`.

  What it is **not**: darkness. A bare hand darkened to gamma 0.25 is still
  detected at 0.97 confidence.

- **The image boost makes detection worse, not better.** Same measurement:
  a bare hand at 100% detection drops to 0% under `gamma 0.55 + CLAHE 2.5`
  (68% at the lowest threshold). Brightening an already well-exposed frame
  blows out the skin texture the model relies on. `--gamma` and `--clahe`
  therefore default to off and should stay off unless a sweep shows otherwise
  for a genuinely underexposed setup.
- One webcam gives **approximate** millimetres (see the scale spread above).
- Occluded fingers are estimated, not measured — the glove's advantage.
- Handedness is unreliable in the egocentric views of the reference dataset,
  for both devices. Anything that mixes hands should use the chirality-aware
  features, and any conclusion that depends on left/right should be checked
  with the mirrored-refit test in `compare_to_tracker.py`.
- `selftest_sync.py` numbers are a plumbing and logic check, not evidence
  about real hardware.
