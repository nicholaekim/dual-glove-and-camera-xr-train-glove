# Ultraleap Stereo IR 170 + XR Trainer glove: integration plan

Written 2026-09-16. Planning pass (Fable). Execution is meant for a separate
session working inside this repo (`non glove xr trainer`).

## 0. What the professor asked for, in one paragraph

Keep the StretchSense glove + XR Trainer pipeline for what it measures well
(per-joint finger flexion, 0.1 degree, works when fingers are hidden), and use
the Ultraleap Stereo IR 170 camera he supplied to capture the things the glove
cannot: hand position and orientation in space, finger spread (abduction),
thumb opposition, crossed fingers, wrist angle, and real bone lengths. Those
are exactly the 21 professor reference poses we marked NA in August.

Deliverable: the camera stream lands in the same 26-joint / 21-keypoint /
professor-txt formats the repo already uses, so every existing tool
(recorder, playback, keypoints21, compare_to_tracker, export_prof_format,
fuse_poses, the LOO classifier) works on camera data without changes.

## 1. Facts established (research, 2026-09-16)

Hardware (Stereo IR 170 evaluation kit, part UL-SIR170-01-C-EV)
- Stereo infrared, 850 nm LEDs, 90 fps, 170 x 170 degree field of view,
  tracking range about 10 to 75 cm from the module.
- We run it on Windows. (The datasheet said Windows only; the download page
  now also lists Hyperion 6.2 packages for macOS, Linux, Raspberry Pi and
  several headsets.) USB 2.0, Micro USB Type-B on the housing, 5 V at
  0.5 A minimum. Use a direct USB port, not an unpowered hub.
- 90 fps is the camera capture rate. The hand-tracking rate is separate
  and can be lower on a slow CPU; read it from the tracking event's
  `framerate` field (or compute it from timestamps) and log it.
- Indoor use. Direct sunlight and other IR sources break tracking.

Software
- The SIR170 download page (https://www.ultraleap.com/downloads/sir170/,
  Operating System = Windows, read directly on 2026-09-16) offers
  Ultraleap Hyperion v6.2.0 (release date 22.09.2025) and a legacy Gemini
  Developer Preview v5.0.0. Stated requirements: Windows 10 64-bit, Intel
  Core i3 5th gen or better with AVX, 2 GB RAM, USB 2.0. Install Hyperion
  6.2.0 first. It bundles the tracking service, Control Panel / visualiser,
  the LeapC SDK at `C:\Program Files\Ultraleap\LeapSDK`, and the OpenXR
  API layer.
- Caveat: Hyperion 6.0 and 6.1 supported only the Leap Motion Controller 2,
  and older support articles (now offline) said the SIR170 needs Gemini V5.
  The 6.2.0 listing on the SIR170 page is the evidence that 6.2 added the
  older devices. If the Control Panel does not list the SIR170 after
  install, uninstall and use Gemini 5.x instead (the 5.0.0 preview on the
  same page, or an archived 5.17 / 5.20 installer). LeapC and the Python
  bindings are the same either way.
- License: the kit is sold for technical evaluation; the Ultraleap SDK
  EULA allows non-commercial and educational use, which covers this
  research as long as nothing ships in a product. Save the EULA text with
  the archived installer.
- Official Python bindings: https://github.com/ultraleap/leapc-python-bindings
  (package `leap`, built on a cffi module `leapc_cffi`). The repo is still
  titled "Gemini" and was last updated in 2023, but its build script only
  needs the LeapSDK layout (`LeapSDK\include\LeapC.h`,
  `LeapSDK\lib\x64\LeapC.dll` and `LeapC.lib`), which Hyperion keeps, so it
  is expected to work with 6.2; that is verified on Day 1, not documented.
  Pre-compiled cffi modules shipped with the SDK cover Python 3.8 only on
  Windows (stated for Gemini 5.17; nothing newer is documented). Any other
  Python version must compile `leapc_cffi` locally (LeapSDK + C compiler;
  `python_requires >= 3.8`, cffi unpinned, and cffi 2.x supports 3.14).
  The script finds the SDK at the default path or via
  `LEAPSDK_INSTALL_LOCATION`, with `LEAPC_HEADER_OVERRIDE` and
  `LEAPC_LIB_OVERRIDE` for odd layouts.
- The OpenXR API layer (XR_EXT_hand_tracking, 26 joints in the same order as
  the glove) is bundled in the tracking package. Not needed for us: it
  requires an OpenXR runtime and app. We consume LeapC in-process instead.
- Company status: Ultraleap sold hand tracking to ROLI in March 2025 and was
  fully acquired in November 2025. docs.ultraleap.com and the downloads page
  are up; support.ultraleap.com did not resolve today. Older releases have
  already disappeared once (Gemini 5.17). Archive the installer and the
  LeapSDK folder in OneDrive the day they are downloaded.

LeapC data model (what the Python `leap` package exposes)
- `leap.Connection()`, `leap.Listener` with `on_tracking_event(event)`.
  `event.timestamp` (microseconds, LeapC clock; pair with `time.time()` at
  receipt), `event.tracking_frame_id`, `event.hands`, `event.framerate`
  (actual tracking rate in Hz; log it, do not assume 90).
- `hand.type` (HandType.Left / Right), `hand.visible_time` (microseconds
  tracked so far), `hand.pinch_strength`, `hand.grab_strength`,
  `hand.pinch_distance`. `hand.confidence` is documented as "not currently
  used (always 1.0)": never gate on it.
- `hand.palm.position`, `.orientation` (quaternion x,y,z,w), `.normal`,
  `.direction`, `.width`, `.velocity`.
- Orientation convention (LeapC struct reference): `LEAP_PALM.orientation`
  is the basis {normal x direction, -normal, -direction}, so local +x points
  sideways across the palm, +y is dorsal (out of the back of the hand) and
  +z points from the fingers back toward the wrist. `LEAP_BONE.rotation` is
  "rotation in world space from the forward direction" with the same
  layout: +z runs from next_joint back to prev_joint (the bone points
  along local -z), +y is that bone's own dorsal direction, +x exits the
  sides of the finger. For a flexed finger the bone's +y is not the global
  -palm.normal; only the palm's +y is. Every rotation is absolute in the
  camera frame, which is what the parent-inverse conversion in Section 3
  expects. Quaternion order is x, y, z, w.
- `hand.digits[0..4]` = thumb, index, middle, ring, pinky. Each digit has
  `bones[0..3]` = metacarpal, proximal, intermediate, distal, each with
  `prev_joint`, `next_joint`, `rotation` (quaternion), `width`.
  The thumb metacarpal is a zero-length bone (Ultraleap convention).
- `hand.arm` is a bone; `arm.next_joint` is the wrist.
- Units: millimetres, microseconds, radians. Right-handed frame, origin at
  the top centre of the module, x along the camera baseline, y up, z toward
  the user (desktop mode). `connection.set_tracking_mode(leap.TrackingMode.Desktop)`.

This machine (checked today)
- Python 3.14.6 is the only interpreter (`py -0`). The repo venv is 3.14.
- Visual Studio 2022 Build Tools with the C++ toolset are installed
  (vswhere finds them), so compiling `leapc_cffi` is possible.
- No Ultraleap software installed yet.

## 2. The gate: does the camera see the gloved hand?

Ultraleap's tracker is trained on bare skin. Gloves are not officially
supported, and our RGB webcam + MediaPipe got 0 of 40 on the gloved hand in
August. Whether the IR 170 tracks a hand inside the black StretchSense glove
decides the architecture, so it is the first experiment after setup.

Protocol (30 minutes, needs only the Control Panel visualiser plus the new
live viewer):
1. Bare hand, desktop mode, 25 to 40 cm above the module: confirm stable
   tracking of both hands, open palm and fist.
2. Same, wearing the glove (both hands, one at a time). Record 20 s each of
   open palm, fist, pinch, spread. Log the fraction of frames in which the
   hand is reported at all, `visible_time` resets (re-acquisitions), and
   fingertip jitter versus the bare-hand run.
3. Mitigations, each as its own 20 s run: a thin white cotton liner glove
   worn over the StretchSense glove; small strips of plain white tape on
   fingertips and knuckles; retro-reflective tape last, because it can
   saturate the IR image. There is no official or community evidence that
   any of these work with the native hand model (the only related report
   is a forum user whose dark tattoo blocked tracking and who suggested a
   white cotton glove), so treat them as experiments and record the result
   either way. Also record the tracking `framerate` in every run.

Evidence to keep from every gate run
- The JSONL and `.lmt` recordings, and the `stats.py` table.
- IR stills of the hand in each condition, saved with
  `scripts/leap/ir_snapshot.py` (to be written on hardware day): it calls
  `connection.set_policy_flags(flags_to_set=[leap.PolicyFlag.Images])`,
  receives `on_image_event(event)` with `event.image[0]` (left) and
  `event.image[1]` (right), reads `image.c_data.properties.width/.height/
  .bpp`, `.data` and `.offset`, copies the buffer inside the callback (LeapC
  reuses it) and writes PNGs. These photos are the professor-facing evidence
  of what the camera actually sees through the glove.

Control Panel settings for the gate (leave everything else at default)
- Tracking mode: Desktop, set explicitly.
- Allow images: on, and confirm `PolicyFlag.Images` comes back active.
- Performance mode: Balanced (default). Do not enable Low Resource.
- Robust / light-robustness and auto-orientation: default; note whether the
  device reports a robust status during the runs (IR interference).
- Interpolation: off for gate metrics. Hyperion's alternate hand models
  (Hand On Object, Microgestures) are HMD-only; do not use them here.

Decision table
- Path A (gloved hand reported in 80 %+ of frames, no more than one
  re-acquisition per 10 s, jitter within 2x the bare-hand value):
  simultaneous capture. Glove supplies flexion, camera supplies palm pose,
  spread, thumb, wrist. `fuse_poses.py` and `fusion.py` apply directly, now
  with metric 3D camera data instead of MediaPipe's normalised guesses.
- Path B (gloved hand not tracked): sequential capture. Record each pose
  twice, glove on and glove off, under one protocol and one file naming
  scheme. Glove files carry flexion, camera files carry everything else.
  The 21 NA poses come from the camera only. No frame-level fusion, but the
  dataset still covers every dimension the professor listed.

Both paths share all of Phase 1 and the analysis tooling, so nothing built
before the gate is wasted.

## 3. Architecture

New package `src/leap_hand/`, shaped like `src/cam_hand/`:

- `stream.py`: `LeapStream` wrapping `leap.Connection` + `leap.Listener`.
  Pushes `LeapHand` snapshots into a queue with the same `drain()` API the
  glove receiver has, so scripts can poll both sources in one loop.
- `types.py`: `LeapHand` dataclass: `hand_side`, `hand_id`, `visible_time_us`,
  `framerate`, `timestamp_us`,
  `frame_id`, `pinch_strength`, `grab_strength`, `palm_pos`, `palm_quat`,
  `abs26` (26 x [x,y,z] in camera mm) and `quat26` (26 x [x,y,z,w]).
- `to_openxr.py`: Leap bones to the 26-joint layout (table below), then
  `absolute_to_relative()` to produce a glove-convention `HandFrame`
  (parent-relative translation and rotation, XYZW quaternions) so
  `forward_kinematics` reproduces the camera positions exactly. Add the
  inverse helper in `xr_hand/kinematics.py` and a round-trip unit test.
- `recorder.py`: `LeapRecorder` writing JSONL with the same keys as
  `xr_hand.recorder._frame_to_dict` plus extras (`source: "leap"`,
  `space: "leap_desktop_mm"`, `hand_id`, `visible_time_us`, `framerate`,
  `pinch_strength`,
  `grab_strength`, `palm_abs`). `FrameRecorder.load` ignores unknown keys,
  so playback, keypoints21 and the exporters read these files unchanged.
- `mock.py`: synthetic `LeapHand` generator (open palm, fist, spread sweep)
  behind the same `drain()` API, so every script runs with `--mock-leap` and
  tests pass on a machine without the camera.

Joint mapping (OpenXR joint <- Leap bone endpoint; orientation <- rotation of
the bone that starts at that joint)

| OpenXR joint | Position | Orientation |
|---|---|---|
| PALM | palm.position | palm.orientation |
| WRIST | arm.next_joint | arm.rotation |
| THUMB_METACARPAL (CMC) | thumb.bones[1].prev_joint | thumb.bones[1].rotation |
| THUMB_PROXIMAL (MCP) | thumb.bones[2].prev_joint | thumb.bones[2].rotation |
| THUMB_DISTAL (IP) | thumb.bones[3].prev_joint | thumb.bones[3].rotation |
| THUMB_TIP | thumb.bones[3].next_joint | thumb.bones[3].rotation |
| F_METACARPAL | f.bones[0].prev_joint | f.bones[0].rotation |
| F_PROXIMAL (MCP) | f.bones[1].prev_joint | f.bones[1].rotation |
| F_INTERMEDIATE (PIP) | f.bones[2].prev_joint | f.bones[2].rotation |
| F_DISTAL (DIP) | f.bones[3].prev_joint | f.bones[3].rotation |
| F_TIP | f.bones[3].next_joint | f.bones[3].rotation |

(F = index, middle, ring, little. Leap's thumb bones[0] is the zero-length
metacarpal, so the thumb chain starts at bones[1].)

Verify on hardware: hold an open palm facing up over the module, run
`viz3d` on the converted frame, and check finger order, chirality and that
fingertips are farthest from the wrist. Then check the rotation convention
from Section 1 numerically: for each bone, `rotation` applied to (0,0,-1)
must point from prev_joint to next_joint within a few degrees. Positions do
not depend on this, and the 21-keypoint tooling uses positions only.

Scripts
- `scripts/leap/live_view.py`: 3D skeleton + HUD (tracking framerate,
  hand ids and re-acquisitions, pinch, grab, hands seen). First smoke test and the tool for the Phase 2 gate.
- `scripts/leap/record_poses.py`: same guided beep protocol and pose list as
  `scripts/glove/record_poses.py` (open_palm, fist, index_point, thumbs_up,
  peace, pinch, three), writing to `recordings/leap/poses/`.
- `scripts/leap/record_frame.py`: professor-frame replication for the camera,
  mirroring the July `record_frame.ps1` flow: show the reference png, count
  down, capture N seconds, save `<frame>_keypoints.txt` in the professor's
  format via the existing exporter path.
- `scripts/record_simultaneous.py --camera leap`: add the Leap backend next
  to the MediaPipe one (Path A only).
- `scripts/leap/stats.py`: per recording, the hand detection rate, number
  of re-acquisitions (hand id changes), mean tracking `framerate`,
  fingertip jitter at rest, and latency (Leap timestamp vs wall clock).
  Used for the gate and the report.

Coordinate policy
- Camera files store absolute camera-frame mm (the real hand) in `abs26`
  and the derived parent-relative HandFrame. Wrist-centred 21-keypoint
  export uses the existing `frame_to_keypoints21(wrist_centered=True)`.
- For glove-vs-camera comparison the canonical frame is the camera (real
  geometry). Rigid alignment only (no scale) when both are metric; keep
  `align.py` scale option for the old MediaPipe files.

## 4. Environment setup (exact steps)

1. Download Hyperion 6.2.0 for Windows from
   https://www.ultraleap.com/downloads/sir170/ . Copy the installer and,
   after install, `C:\Program Files\Ultraleap\LeapSDK` into
   `OneDrive\Desktop\xr trainer\reference\ultraleap\`.
2. Install, plug the camera into a direct USB 2.0/3.0 port, open the
   Ultraleap Control Panel, confirm the device is listed and the visualiser
   shows hands. Set tracking mode to Desktop. Note the firmware/software
   versions in the README.
3. Python bindings. Try the current 3.14 venv first:
   ```
   git clone https://github.com/ultraleap/leapc-python-bindings reference/leapc-python-bindings
   cd reference/leapc-python-bindings
   pip install -r requirements.txt build
   python -m build leapc-cffi
   pip install leapc-cffi/dist/leapc_cffi-0.0.1.tar.gz
   pip install -e leapc-python-api
   python examples/tracking_event_example.py
   ```
   If the cffi build fails on 3.14, install Python 3.11
   (`winget install Python.Python.3.11`), create `.venv-leap` from it, and
   repeat. Set `LEAPSDK_INSTALL_LOCATION` if the SDK is not in the default
   path. Pin whatever works in `pyproject.toml` as an optional extra `leap`.
4. Smoke test: `tracking_event_example.py` prints hand ids and palm
   positions at about 90 Hz.

## 5. Phases, tasks, acceptance

Phase 0. Setup and archive (half a day). Done when the example script prints
frames and the installer + SDK are archived.

Phase 1. Backend (2 to 3 days).
- `leap_hand/{types,stream,to_openxr,recorder,mock}.py`,
  `kinematics.absolute_to_relative`, tests: mapping table, round trip
  (absolute -> relative -> forward_kinematics within 1e-6 mm), recorder load
  via `FrameRecorder.load`, keypoints21 on a mock frame.
- `scripts/leap/live_view.py` and `scripts/leap/stats.py`.
- Done when a 10 s live recording plays back in `playback.py` and exports
  through `export_keypoints21` and `export_prof_format` with no code changes
  to those tools.

Phase 2. Gate experiment (half a day). Run Section 2. Write
`results/leap_glove_visibility.txt` with per-condition detection rate,
re-acquisitions, tracking framerate and jitter. Choose Path A or B and
record the choice in the README.

Phase 3. Data collection (1 to 2 days of recording).
- Bare hand, both hands, 7 poses x 3 takes: `recordings/leap/poses/`.
- The 21 NA professor frames plus a re-record of a 10-frame subset of the
  81 DONE frames (for camera-vs-glove-vs-tracker on the same poses):
  `recordings/leap/prof_frames/`.
- Dimension probes the glove cannot do, 10 s each: abduction sweep
  (fingers together to max spread), thumb opposition circle, crossed index
  over middle, wrist flexion/extension and radial/ulnar deviation, hand
  moving in a 30 cm box while holding a fist.
- Path A only: simultaneous glove + camera takes into `recordings/sync/`.

Phase 4. Analysis and report (1 to 2 days).
- LOO nearest-centroid on 21 keypoints: camera-only rows next to the July
  and August glove rows (same script, `analyze_poses_cam.py` generalised to
  read leap files).
- `compare_to_tracker.py` on the 21 NA frames and the 10-frame subset:
  RMSE after rigid alignment, chirality agreement. This is the first time
  the NA poses get any number at all.
- Dimension probes: report the measured abduction range per finger, the
  thumb opposition angle range, and wrist angle range, against the glove's
  fixed 5/10/15 degree splay and zero wrist motion.
- Stability: jitter sigma at rest, dropout rate, end-to-end latency
  (Leap timestamp vs wall clock).
- Path A: re-run `fuse_poses.py` with metric camera data; report the
  three-row glove / camera / fused table.
- Update README and the progress report PDF generator with the new section.

Phase 5 (optional, after the professor sees Phase 4). Real-geometry glove:
measure bone lengths from a camera calibration pose and apply the glove's
flexion angles to those lengths, so glove skeletons stop using the XR
Trainer template hand.

## 6. Recording protocol notes

- Module flat on the table, lens up (desktop mode), centred in front of
  the user, at least 10 cm from any monitor edge, matte dark surface. No
  window light or overhead spotlights on the hands, no other IR emitters
  (depth cameras, some webcams). Wipe the lens before each session.
- Hand 20 to 50 cm above the module, palm roughly facing the camera for
  spread and thumb poses; the previous July/August protocols apply
  otherwise (announce pose, countdown beeps, no keyboard while gloved).
- Left/right: record both hands. Compare the camera's `hand.type` against
  the professor's labels to close the mirrored-label question from August.
- Gate recordings on presence, not confidence (the field is a constant
  1.0): a frame counts when the hand is reported, `visible_time` is at
  least 0.3 s, and the hand type has not flipped in the last 0.5 s.

## 7. Risks

- Gloved hand invisible to IR tracking (most likely). Path B covers it.
- Hyperion 6.2 does not enumerate the SIR170 on this PC. Gemini 5.x
  fallback (Section 1); same code.
- Python 3.14 cffi build fails. Fallback to a 3.11 venv (Section 4).
- Downloads move or vanish after the ROLI transition. Archive on day one.
- USB power: the module needs 0.5 A; hubs and some laptop ports brown out.
- Handedness flips at the edge of the field of view. Confidence gating and
  `visible_time` handle it; keep hands inside 10 to 60 cm.
- Convention mistakes in the 26-joint mapping. The visual check and the
  round-trip test catch them before any data is collected.

## 8. Corrections to the ChatGPT "Deep Research" report (Downloads folder)

- LeapC function names are wrong there (`LeapC_OpenConnection` does not
  exist; the API is `LeapCreateConnection`, `LeapOpenConnection`,
  `LeapPollConnection`). Irrelevant for us because we use the Python package.
- It says the glove suffers from IMU drift. The glove has no position or
  orientation at all; wrist and palm never move in the stream.
- It recommends Gemini 5.2+. The current download for the SIR170 is
  Hyperion 6.2.0.
- It suggests re-injecting camera data over OSC to port 9002. Unnecessary:
  the camera is consumed in-process and written straight to JSONL.
- Its repo review was inferred, not read, and its 3-month roadmap is
  over-scoped for this work; Sections 3 to 5 above replace it.

## 9. Verification log (2026-09-16)

The plan was cross-checked in the ChatGPT project thread "Research camera
integration approach" (xr training work study project), and against the
official pages directly. Outcome:

- Confirmed by the official SIR170 download page and by ChatGPT after
  browsing: Hyperion 6.2.0 (22 Sep 2025) is offered for the SIR170 on
  Windows; the same version is offered for the original Leap Motion
  Controller. ChatGPT's first pass (done without browsing) claimed the
  opposite and withdrew it. No first-party 6.2.0 changelog was found, so
  the Day 1 device check in Section 1 stays.
- Confirmed: OpenXR joint order (PALM 0, WRIST 1, thumb 2 to 5, fingers 6
  to 25), the PALM and WRIST rows of the mapping table, the zero-length
  Leap thumb metacarpal, and the LeapC frame (right-handed, x along the
  baseline, y up, z toward the user).
- Corrected in this plan: bone and palm axis conventions (Section 1),
  `confidence` is a constant 1.0 and was removed from every gate, the
  tracking rate is read from the event rather than assumed to be 90 Hz,
  the "Windows only" statement was softened.
- Confirmed: desktop mode is the supported orientation (lenses up, centred
  in front of the user), preferred depth 10 to 75 cm, keep displays and
  other IR sources out of view, and the Control Panel shows IR
  interference.
- Confirmed from the bindings source: `TrackingEvent` exposes
  `timestamp`, `tracking_frame_id`, `hands` and `framerate`; the cffi build
  script needs `LeapC.h`, `LeapC.dll`, `LeapC.lib` and honours
  `LEAPSDK_INSTALL_LOCATION`. No documented cffi or MSVC blocker for Python
  3.12 to 3.14, and no documented success either.
- Still empirical, nobody has data: IR tracking of the black StretchSense
  glove, whether liner gloves or tape help, whether Hyperion 6.2 enumerates
  this particular unit, whether the 2023 Python bindings build against the
  Hyperion SDK, and whether `leapc_cffi` builds on Python 3.14. All five
  are covered by Phase 0 and the Phase 2 gate, and each has a fallback.

Phase 0 outcome, same day (evening of 2026-09-16)
- Hyperion installed: build string `6.2.0+2025.07.11.824112c4.CI1622376`,
  service key `UltraleapTracking`, display name "Ultraleap Tracking
  Service", running. SDK at the default path with `LeapC.h`, `LeapC.dll`,
  `LeapC.lib`. Installer, SDK copy, the bindings clone and the built wheel
  are archived in `..\xr trainer\reference\ultraleap\`.
- `leapc_cffi` builds on Python 3.14 with the VS 2022 Build Tools
  (`leapc_cffi-0.0.1-cp314-cp314-win_amd64.whl`). The 3.11 fallback is not
  needed. Two of the five open items are therefore closed.
- Lesson: the bindings' `requirements.txt` pulls `opencv-python`, which
  shares the `cv2` folder with mediapipe's `opencv-contrib-python`;
  installing it, or uninstalling it afterwards, breaks `cv2`. The setup
  script now filters it out; the repair is
  `pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93`.
- Design review before coding (ChatGPT, browsing the bindings source):
  compare `hand.type` with `leap.HandType` by isinstance then enum, use
  `connect()`/`disconnect()` for a long-lived stream, `leap.get_now() -
  event.timestamp` is frame age at receipt, `leap.Recording`/`leap.Recorder`
  give raw `.lmt` capture and replay, detect the service by display name.
  All applied in `src/leap_hand`.
- Still open for hardware day: device enumeration (no camera was attached
  when the checker ran), the visual and numeric convention checks, and the
  glove gate itself.

## 10. Handoff block for the executing session

Context: repo `C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer`
(GitHub nicholaekim/dual-glove-and-camera-xr-train-glove). Glove code in
`src/xr_hand`, MediaPipe camera code in `src/cam_hand`, comparison scripts
in `scripts/`. Frame convention: `xr_hand/joints.py` (26 OpenXR joints,
parent-relative XYZW), `xr_hand/kinematics.py` (FK), `xr_hand/keypoints21.py`
(26 -> 21). Read `src/cam_hand/fusion.py` docstring for the fusion design.

First commands: Section 4. First code: Section 3 in the order types,
to_openxr + tests, mock, stream, recorder, live_view. Do not touch the
glove receiver or the existing exporters; they must keep working unchanged.

Definition of done for the first PR: `pytest` green with the new tests, a
mock recording plays back in `scripts/glove/playback.py`, and
`scripts/leap/live_view.py` shows a live hand at about 90 Hz.
