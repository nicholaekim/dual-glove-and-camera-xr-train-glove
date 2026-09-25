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
- Hyperion 6.2.0 installed the same evening (see the Phase 0 outcome in
  Section 9); the notes above describe the machine before that.

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
  `scripts/leap/ir_snapshot.py` (written 2026-09-16; `scripts/leap/gate.py`
  takes them automatically during each condition): it calls
  `connection.set_policy_flags(flags_to_set=[leap.enums.PolicyFlag.Images])`
  — the flag lives in `leap.enums`, **not** on the package root:
  `leap.PolicyFlag` is an AttributeError (checked against the installed
  bindings, 2026-09-16) —
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

Phase 2. Gate experiment (half a day). Run Section 2 with
`scripts/leap/gate.py`, which writes `results/leap_gate/REPORT.txt`:
per-condition detection rate, re-acquisitions, tracking framerate and jitter,
then the Path A / Path B verdict in the report's own words. Record the choice
in the README.

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

Phase 1 outcome (same evening)
- `src/leap_hand` (types, to_openxr, stream, recorder, mock, replay, stats),
  `scripts/leap/{check_setup.py, setup_bindings.ps1, live_view.py,
  record_poses.py, stats.py}` and `tests/test_leap_hand.py` written by an
  Opus session from Section 3, then code-reviewed by ChatGPT against the
  bindings source. Review found four real bugs (tracking-mode failure
  swallowed, frames accepted before Desktop mode was set, a failed connect
  left the LeapC connection open, an invalid device-info fallback) plus
  wording and robustness items; all eleven were fixed. 81 tests pass; the
  mock recording plays back and exports through the unchanged glove tools.
  Merged to main on 2026-09-16.
- Measured with the service running and no camera: `get_tracking_mode()`
  times out because the TrackingMode event only arrives once a device is
  present. The stream therefore sets Desktop mode right after connect
  (fatal on failure), clears its queue, waits for the device and first
  frame, then confirms the mode and refuses to run if it is not Desktop.
- `HandFrame.timestamp` for camera frames is the Leap clock in seconds
  (the glove's is a device tick counter); `wall_time` in the JSONL is the
  shared clock used for pairing, and `timestamp_us` is kept alongside.

Hardware day (2026-09-16, later the same evening, camera attached)
- Hyperion 6.2.0 enumerates this unit: serial
  `LE3100040000001DD2-3610-2217501E20`, reported as `DevicePID.SIR170`,
  Windows shows it as a USB composite device (VID 2936, PID 1202) with a
  camera interface. Streams at 89 to 90 Hz tracking rate with the mode
  confirmed as Desktop. The first checker run right after plug-in saw zero
  events in its 3 s window; the device needs a few seconds to start.
- Numeric convention check on a bare hand (587 hand samples, both hands):
  `rotation * (0,0,-1)` matches prev_joint to next_joint to 0.00 degrees on
  every bone; thumb `bones[0]` length 0.000 mm; palm basis matches
  {normal x direction, -normal, -direction} to 0.00 degrees; palm to wrist
  72 mm; fingertips farther from the wrist than the knuckles on all five
  fingers (wrist to tip: thumb 117, index 141, middle 140, ring 131, pinky
  117 mm). Frame age at receipt median 9.5 ms, p95 10.4 ms. Every
  documented convention in Section 1 is therefore confirmed on hardware;
  the mapping needs no change.
- Closed: device enumeration, bindings against the Hyperion SDK, the
  convention checks. Still open: the glove gate (Section 2) and the raw
  `.lmt` capture, which has not yet run with a device.

Gate result (2026-09-16, 22:47 to 22:52, one operator, left hand)
- The IR camera tracks the hand inside the black StretchSense glove. Glove
  run: 1672 frames in 18.6 s, 98.6 % detection, 0 re-acquisitions, 0.20 mm
  fingertip jitter at 90 Hz. Bare run, same hand: 79.2 % detection as
  reported (about 89 % against the 90 Hz tracker rate; the report's
  denominator used the file's own cadence, which timestamp jitter inflated
  to 101 Hz, a bias fixed in the next tooling pass), 1 re-acquisition,
  0.54 mm jitter. The right hand only drifted into view briefly.
- Why it works: in the 850 nm IR stills the glove renders bright, close to
  white, with the sensor grid visible as dots, so the fabric reflects near
  IR strongly even though it is black to the eye. Every glove still has a
  sidecar confirming the tracker reported the hand at that instant. The
  stills contain the operator's face and stay in the ignored `recordings/`
  tree.
- Verdict: Path A, simultaneous glove + camera capture, provisionally.
  Second opinion (ChatGPT, browsing): supported but not yet demonstrated;
  before it goes to the professor, add a centred right-hand glove run, both
  hands at once, fist and pinch and open/spread at about 20, 35 and 50 cm,
  and a repeat on a second day. Do not attribute the better-than-bare
  numbers to the fabric; black textiles vary widely in near-IR reflectance
  and the difference is within run-to-run variation until paired repeats
  say otherwise.
- Raw `.lmt` capture ran with a device for the first time (empty scene,
  322 events, reads back through `leap.Recording`).
- Recomputed with the tracker-rate denominator (`gate.py --recompute`):
  bare left 89.2 %, glove left 100.0 %, verdict unchanged.

Path A tooling (merged 2026-09-16, 130 tests)
- `scripts/record_simultaneous.py --camera leap`: one guided session records
  the glove over OSC and the camera together on a shared wall clock. The
  camera keeps every frame (two independently throttled recorders drift
  apart and lose two thirds of the pairs); the glove keeps its `--hz`.
- `scripts/fuse_poses.py` and `cam_hand/fusion.py`: leap files are detected
  by `source: "leap"` and aligned rigidly, no scale, because they are
  already metric. The MediaPipe path is unchanged. On the known-answer
  self-test expressed as leap files: glove 24/36, camera 36/36, fused
  36/36.
- `scripts/leap/record_frame.py`: camera version of the professor-frame
  replication (medoid of a short take, professor txt format in mm).
- Not yet run with a real hand or with XR Trainer streaming.

Second gate session (2026-09-17, 13:19 to 13:23, daytime)
- Report: bare right 62 %, glove_right 60 %, glove_both 75 / 80 %,
  glove_20cm 94 / 100 %, glove_35cm 92 / 87 %, glove_50cm 51 %. Verdict
  still Path A (glove_20cm and glove_35cm meet all three thresholds, with
  both hands in view at once).
- Reading the losses from the recordings: they are single long blocks of
  1 to 8 s, not flicker. Bare hand: lost for 5 s when the hand turned
  edge-on to the camera (29 % of its frames were edge-on). glove_right: the
  hand sank to 99 mm above the lens, the bottom of the 10 to 75 cm range,
  and was lost for 8 s. In four gloved runs the loss began while the hand
  was a fist (`grab_strength` 1.0 at the last tracked frame). No bare-hand
  fist was recorded in this session, and the first session's gloved fist
  tracked at 100 %, so whether fists are harder with the glove is open.
- The distance labels were not achieved: median palm height was 196 to
  207 mm for "20cm", 198 to 240 mm for "35cm" and 267 mm for "50cm". There
  is no evidence yet about tracking beyond about 30 cm.
- Conclusion: the glove did not track worse than bare skin in the same
  session; the session's protocol was too loose to say more. The gate
  runner needs timed pose prompts, a live height readout against a target
  band, and per-pose detection rates with a paired bare run, so that a
  fist with the glove is compared with a fist without it.

First real Path A session (2026-09-17, 13:44, both gloves, XR Trainer
streaming, 6 poses x 1 take x 5 s, both hands over the camera at once)
- A first attempt recorded nothing: the script waited for a tracked hand
  while its HUD printed "0.0 Hz", which read as a dead camera. The HUD now
  says "camera: running, NO HAND IN VIEW", tells the operator to hold the
  hand over the lens, and beeps until it sees one.
- Plumbing: paired on `capture_time`, 193 of 242 glove frames matched within
  50 ms. Where the camera had a hand it held it for the whole take: 100 %
  detection, 0 re-acquisitions, 0.10 to 0.93 mm jitter, frame age 10 to
  12 ms, including a gloved FIST (right hand, grab 1.00, 449 of 450 frames).
  Palm-fit diagnostic 4.7 mm median: the template hand versus the real one.
- Coverage gaps: `peace` has no camera frames at all, and the left hand is
  missing from the camera in `fist` and `thumbs_up`. With two hands in view
  each sits 8 to 10 cm off centre.
- What each sensor saw, from the features: the glove's `pinch` is
  numerically identical to its `open_palm` (as in July), while the camera
  shows the index curling (1.76 to 1.27) and the thumb-index gap closing
  (0.87 to 0.35): the camera supplies exactly what the glove lacks. The
  reverse holds for `thumbs_up`: the glove has the four fingers curled
  (0.66) but the camera, looking at an edge-on hand, reports them nearly
  straight (1.69, grab 0.04). Each sensor is wrong where the other is right.
- Classifier on this session: glove 9/10 (misses pinch), camera 6/8, fused
  5/10. The fused number is not meaningful yet: one take per pose leaves one
  training sample per class, and classes mix fused and glove-only samples
  where the camera lost a hand. It does expose a real design gap: fusion
  takes spread and thumb direction from the camera whenever a camera frame
  exists, even when the hand is edge-on or the glove says the finger is
  fully curled, which is when the camera is guessing.

Day-1 session with the corrected recorder (2026-09-18, 18:36 to 18:48,
6 poses x 5 takes x 2 hands, one hand at a time; `recordings/sync_day1`)
- 59/60 takes. fist_left take 1 failed all four attempts: the tracker lost
  the gloved left hand for 0.5 s right after it closed. Ten rejected
  attempts in all (six left fists, two left thumbs up, two left pinch), all
  "hand lost" or "re-acquired", kept under `rejected/`. Right hand: none.
- Pose check: 39 ok, 20 warn (one sensor disagreeing), 0 mismatch. So the
  previous session's label errors were the missing pose name, as suspected.
- Glove at 60 Hz, 300 frames per take, worst gap 33 ms, no dropout in about
  12 minutes. Pairing 17689/17695 within 50 ms, camera used 97 %, palm fit
  3.0 mm median. One IR still per take in `stills/`.
- Per DOF (median over takes): curl agrees between sensors on the left hand
  within about 0.2 (template offset). The RIGHT glove under-reads ring and
  pinky curl in peace, thumbs up and index point (pinky 1.55 to 1.68 where
  the camera says 0.78 to 0.90, consistent across takes; right pinky in a
  fist 1.28 vs left 0.77). Pinch: camera index 1.16 to 1.39 and tip gap
  0.11 to 0.28 in 10/10 takes, glove exactly on its rail in 10/10. Spread
  from the glove is a template constant (5.96 / 12.89 / 7.92 on every open
  palm); the camera gives 19 to 24 / 2 to 4 / 17 to 19 deg with a few
  degrees of take-to-take spread. Right thumbs up: the thumb agreement gate
  rejects a correct camera on every frame because the glove's ring/pinky
  are wrong. The "spread thumb-index" (base bone) row produces fused values
  outside both inputs (a hybrid-skeleton artifact).
- Leave-one-take-out classifier: glove 43/59, camera 59/59, fused 56/59;
  the three fused misses are right-hand takes where glove-owned curl is
  wrong. Reported as is: fusion needs per-hand, per-finger reliability
  logic; glove curl does not automatically dominate.

The disagreement override (2026-09-25, main 2bc865c+, 441 tests): the
fusion no longer follows a glove that has drifted
- What prompted it: a live left-hand run (`recordings/demo/2026-09-25_1037_left`)
  in which the glove registered a fist at 8 s, then from 22 s reported the
  fingers mostly open (index 1.73, middle 1.40, ring 1.64, pinky 1.36 on
  rails of 1.74/1.82/1.68/1.44) while the camera, trusted and palm-facing,
  saw a fist. The fused hand followed the glove because the glove owned
  curl and the rail-only override needs the glove bit-exact on its rail.
- `RailOverrideParams(mode="disagree")`: a frame qualifies for a finger when
  the camera frame passes the trust gates and the view gate, both sensors
  are normalisable on the warm-up endpoints, and camera fraction minus
  glove fraction >= 0.35 (camera more bent; `disagree_both_ways` adds the
  reverse, off by default). Enter after 10 consecutive frames, release
  below 0.20 after 5. On all four fingers of both hands. When active the
  camera takes the finger's curl exactly as the rail override does. The
  thumb vote's `disputed` is unchanged.
- Measured (profile on, anchor off):
    day 1: fused 57/59 -> 57/59; false fires 0.66 % of open-palm and fist
           finger-frames; right pinky residual .245 -> .190
    day 2: fused 53/60 -> 59/60; false fires 0.03 %; right ring .195 ->
           .123, pinky .400 -> .138, middle .068 -> .051
    day 3: fused 35/36 -> 35/36; false fires 1.28 %; right pinky .071 ->
           .045, everything else within 0.005
    both-ways on day 2: 58/60, so camera-more-bent only is the default.
- Now the profile default (`override_mode: disagree` in profiles/default.json
  and the NK profile); `--override-mode rail` restores the old behaviour.
  The live path (fuse_live.py, demo.py) is being wired to read the same key
  so the demo behaves like the offline tool.
- Demo honesty (same commit): the camera panel dims with the reason when
  its frame is not trusted, fingers whose fractions differ by 0.35 or more
  get an amber ring (green when the camera took over) and a `disagree:`
  badge, and the saved JSONL now carries each step's glove and camera
  inputs.
- Unchanged by this: the recalibration verdict (still off). The override
  acts per frame on a trusted camera; it learns nothing across sessions.

Live fusion on hardware (2026-09-24, main 64c3678, 417 tests): works
- `scripts/demo.py --live` (same path as `fuse_live.py`, plus the three-hand
  window) ran on both gloves, one hand per run. Left, 60 s: 3558 fused
  frames at 59 Hz, 68 % paired with the camera, thumb from the camera on
  60 % of frames, spread on about half, template fit saved. Right, 30 s:
  acquired in 6 s, template fit on 360 open-palm frames, 1810 fused frames,
  93 % paired, thumb from the camera 96 %, spread 70 to 84 %, index
  override on 0.5 %. Endpoints learned in the 8 s warm-up are sensible
  (glove spans 0.7 to 1.2, camera 0.6 to 1.0).
- In a `--hand both` run the right hand was refused during the warm-up
  and only the console knew why; the warm-up report and the end summary
  are now saved as `<out>.warmup.txt` and `<out>.summary.txt`. For a demo,
  run one hand at a time.
- The recorded demo (`--replay recordings/sync_day2`) plays 60 takes with
  a pose guess right on 86 % of frames (54 of 60 takes by majority); the
  right glove's index point and peace are the misses. `docs/demo.md` is
  the runbook.

Day 3 (2026-09-23 evening, `recordings/sync_day3`): the recalibration's
prospective test, and it fails the per-finger criterion
- Session: 3 takes x 6 poses x 2 hands = 36 takes, 8 rejected attempts
  kept, 0 label mismatches (7 warnings), glove 60 Hz, camera 90 Hz. Gloves
  recalibrated in XR Train before the session. Nothing from this session
  was used to fit anything.
- Baseline: glove 30/36, camera 36/36, fused with profile 35/36. The right
  glove behaved well today: before any correction its trusted-frame
  residuals were 0.03 to 0.07 (day 2: up to 0.40 on the pinky). So the
  right glove's "reads too open" fault is SESSION-DEPENDENT, present on
  days 1 and 2 and largely absent on day 3.
- Cross-session recalibration applied to day 3 (coefficients from day 1,
  endpoints re-learned): fused 36/36 (day 2 model: 35/36; own from day 1:
  36/36). The classifier hides what the per-finger residuals show. Before
  -> after, median |camera - glove fraction| on trusted frames, day 1 model:
  left index .048 -> .027, middle .038 -> .055, ring .036 -> .043, pinky
  .147 -> .042; right index .038 -> .066, middle .032 -> .123, ring .029
  -> .134, pinky .071 -> .109. Per pose on the right hand it helps where
  day 1's fault reappears (index_point middle .48 -> .16, pinky .53 -> .12;
  peace pinky .38 -> .08) and harms where the glove was right (pinch middle
  .04 -> .77, thumbs_up index .04 -> .32 and ring .01 -> .17, fist all four
  worse). The day 2 model is worse still (right middle .032 -> .163).
- Verdict: the reviewer's "no meaningful per-finger regression" criterion
  fails, so `--recalibrate` stays OFF and is not stored in the profile. A
  calibration fitted on a day the glove misbehaved over-corrects on a day
  it does not; the right glove's OBSERVED calibration error is
  session-dependent (reviewer wording), so a static model cannot remove
  it. Reviewer verdict (2026-09-23): default off and nothing stored in
  the profile is supported; the four checks asked for hold by
  construction (one HandScale normalises before and after, the rail rule
  runs at fit and at apply and is tested, one finger order throughout,
  residual columns computed on the same trusted rows). The next
  legitimate experiment is a within-session adaptive correction learned
  only from the current session's trusted camera frames; a 6 s open/fist
  warm-up cannot fit a five-input model (two poses only). What would still be legitimate: a correction
  learned WITHIN the session from trusted camera frames (the anchor's idea,
  with the rail rule and per-pose care), or the fault fixed at the source
  (StretchSense: raw stream, calibration sets).
- The live frozen-warm-up run (`fuse_live.py`) was attempted twice; the
  first attempt started its 3 s phases before a hand was over the module
  and was refused (camera 0 frames, glove span 0). The script now has an
  ACQUIRE gate, a countdown and per-hand warm-ups (main 6a3b742); the
  second attempt left no output file, so the frozen-warm-up path is still
  untested on hardware. With the offline verdict above it is no longer the
  deciding test.

The professor's 21 poses, recorded with the camera (2026-09-23, main
dc6821f, 397 tests)
- All 21 `_NA` frames recorded bare-handed (left hand) with
  `scripts/leap/record_prof_frames.py --seconds 3 --prep 3`; keypoint files
  in his format under `recordings/leap/prof_frames/` and copied into each
  `frame_<id>_NA/camera_recording_LEFT/` plus a hand-in folder
  `xr trainer\camera poses for professor 2026-09-23\` (README, table, CSV).
- Scored with the new `scripts/leap/compare_prof_frames.py` (Umeyama on the
  wrist and knuckles, error over all 21 landmarks, mirrored refit as in
  `compare_to_tracker.py`): median 17.6 units after the better fit (22.3
  direct), solved scale median 1.01 (0.79 to 1.34), so his units are
  millimetres at this hand size. Worst landmarks: fingertips and thumb.
- Chirality: the operator used the left hand throughout; the Ultraleap
  labelled it "right" in 20 of 21 frames and his tracker labelled the same
  arm "right" in 10 of 21 (153624 right, 156023 left, same arm and chair).
  The mirrored fit wins exactly when the labels differ, so a mislabel is a
  mirror image and the files cannot tell the physical hand. Both trackers
  are unreliable on handedness for hanging or edge-on hands.
- Frame 204909 re-recorded once (39 -> 19). Frame 252114 (peace, edge-on)
  stays poor after two attempts (index tip 96 off): the camera assigns the
  extended fingers to the wrong chains from that angle.
- Tool fixes on the way: the runner's done-check looked inside the frame
  folder while the keypoint file is written next to it (re-recorded 8
  frames for nothing); `--hand left` dropped every frame of the second run
  because the tracker's label was "right", so the runner now passes no hand
  filter by default, `write_prof_file` keeps the majority-label block, and
  `--rewrite` regenerates the keypoint files from the newest take without
  recording. `record_frame.py` opens the reference photo on screen.

Five code fixes and what they measured (2026-09-23; 392 tests; reviewed
in a fresh ChatGPT chat in the project, its corrections adopted)
- Camera drift anchor (`DriftAnchor`, `--drift-anchor`, default OFF,
  experimental). A complementary filter: the residual camera fraction minus
  glove fraction, learned on trusted frames only (view under 50 deg, stable
  id, off rail, not disputed), median over 5 s, applied to every frame by
  bending the glove finger. Done properly (gates read the uncorrected
  curls, `gate_curls`) it gains at most one take per day: day 1 57 -> 57 or
  58 of 59, day 2 53 -> 53 or 54 of 60, and the day 1 gain was an artifact
  of a knuckle-only bend. Per pose, the right hand's median residual spans
  0.4 to 1.05 fraction units (right middle: -0.19 in fist, +0.86 in index
  point) while the change within a 5 s take is 0.02 to 0.10. Reviewer
  wording adopted: pose dependence dominates the glove's error over the
  timescale of these takes; creep may coexist. The anchor now resets at
  every take boundary and per hand (`reset(hand)`), keeps constant memory,
  and the bend is spread over the three joints (50/30/20).
- Camera-referenced glove recalibration (`GloveRecalibration`,
  `--recalibrate own|cross`, default OFF). Per hand, camera fraction of
  finger i = a_i + sum_j b_ij * glove fraction j over all five glove
  fractions (`cross`) or its own only (`own`), ridge on standardised inputs
  (lambda 0.01), trusted frames only. Rail rule: a finger's own railed
  readings neither teach nor get corrected (a rail means anything from
  straight to flexed; that dispute is the rail override's), while railed
  neighbours stay valid inputs. Without the rule the index was poisoned by
  pinch (open palm residual 0.01 -> 0.26). Evaluated leave-one-take-out for
  the calibration, then the usual take-level classifier:
    day 1: ordinary 55, profile 57, profile + own 57, profile + cross 57 / 59
    day 2: ordinary 51, profile 53, profile + own 59, profile + cross 59 / 60
  Held-out residual (median |camera - glove fraction|), cross: right pinky
  .245 -> .044 (day 1), .400 -> .062 (day 2); right ring .087 -> .026,
  .195 -> .093; right middle .027 -> .015, .068 -> .041; left middle the
  only finger slightly worse. Per pose (right hand, cross, day 2):
  index_point middle .86 -> .05, pinky .83 -> .25; peace ring .46 -> .16,
  pinky .66 -> .14; thumbs_up pinky .46 -> .10; open palm and pinch
  untouched (railed). Whole-session hold-out (`--recalibrate-from DIR`,
  coefficients carry, endpoints re-learned on the applied session): fitted
  on day 1 and applied to day 2, cross 59/60 (profile 53/60); fitted on day
  2 and applied to day 1, cross 57/59 (profile 57/59), own 55/59. Day 2's
  remaining miss is pinch_right_take1 (read as peace); day 1's two misses
  are the pinch takes (glove index on its rail, read as open palm).
  Default stays off. Reviewer wording adopted: the error is not explained
  by a single offset, and the other channels' state improves the
  correction, which is consistent with channel cross-talk or a
  pose-dependent calibration error (`own` also recovers day 2 when fitted
  on day 1, so this is not proof of cross-talk; that needs one finger
  moved with its neighbours held still). The reviewer no longer asks for
  a second operator before enabling it for this glove and operator; the
  decisive test is a prospective frozen-calibration session (the live
  warm-up fits the model, it is frozen, a new session that played no part
  in development is scored), plus no reliable finger getting materially
  worse, a bypass when live inputs fall outside the fitted range, and the
  same benefit after don/doff. Other gloves or operators start from off.
- Glove lag from the profile (`glove_lag_s`: left 0.10, right 0.47 s,
  from the finger sweeps; used only when the session's own clips cannot
  measure one; `--glove-lag none` still applies nothing). No measurable
  classifier effect in these held-pose tests; about 5 % fewer paired frames
  at take edges because a glove frame stamped in the first 0.47 s of a take
  describes the hand before the camera recording began. Not an intrinsic
  glove constant: transport and software latency can change with machine
  load or XR Train version.
- Rail override on all four fingers (`--rail-fingers`, evaluation only):
  no classifier row changes on either day; middle never fires, ring 6
  frames (day 2), pinky 5 frames (day 1) and 906 frames, 6 % (day 2).
  Profile stays index only. Pinky is the next candidate: needs a repeated
  sweep and an override precision count (when the rail fires, how often is
  the camera's flex actually right).
- IR stills cropped to the tracked hand (`camera_view.py --still hand`,
  default): the projected skeleton's box, 35 % margin, at least 192 px,
  slid inside the frame; no hand tracked, no file, and a
  `<stem>.skipped.txt` note gives the reason, which the recorder writes to
  meta.json as `still_missing_reason` (`unknown` when there is neither).
  Reduces what the still shows; a hand held near the face can still include
  part of it; the file stays out of git.
- Live fusion (`scripts/fuse_live.py`, `src/cam_hand/live_fusion.py`):
  6 s coached warm-up (open palm 3 s, fist 3 s) learns rails, endpoints and
  the template measurement (an initial calibration, not the offline
  learning over a session); refuses a hand whose open/fist separation is
  under the gate minimums; pairs each glove frame with the camera frame at
  t minus the profile's lag within 50 ms; runs `fuse_all`'s per-frame
  sequence; outputs JSONL and OSC (`/fused/<hand>/keypoints21`, 63 floats).
  Replay equivalence (`--replay recordings/sync_day2`): all 60 takes agree
  with the offline fusion, largest point difference 0, no source
  mismatches, same paired counts. Before trusting a live run: replay a
  recorded session, repeat the warm-up across don/doff and days, check the
  lag sign with one open-to-fist movement.

What XR Train's own files show (2026-09-22, read from the local install:
`AppData/LocalLow/StretchSense/XR Train/Player*.log`, the registry
PlayerPrefs and strings in `StretchSense.CompanionApp.Runtime.dll`)
- XR Train logs the glove's own data gaps: "High delta times detected"
  appears 26 times in the previous session (43 gap values, median 0.48 s,
  largest 3.3 and 3.9 s) and twice in today's. So XR Train itself sees the
  sample gaps at its input; the source could be the glove, the BLE/OS
  transport or XR Train's input handling (reviewer wording).
- The runtime contains exponential-weight and rolling-average smoothing of
  the sensor values (`SmoothCapacitancesWithExponentialWeights`,
  `SmoothRollingAverage`, `_smoothingFactor`, `_smoothingBufferSize`), with
  no user setting in this build. Could contribute to latency; whether the
  filters are active on this path, or explain the per-hand difference, is
  not measured (reviewer wording).
- Feature flags at startup: Pinch = False, FingerSplay = False,
  PinkyTouch = False, EarlyAccess = False, Internal = False. The runtime
  holds a finger-splay model ("Cannot train Splay because the FingerSplay
  feature flag is off", "Splay Model Post Processing") and a pinch feature
  ("The Pinch feature is not enabled in this build"), behind an INTERNAL
  SETTINGS panel that this build hides. The articulation manager also
  lists "Advanced", "Splay" and "Basic Splay" calibration sets beyond
  "Basic". Worth asking StretchSense whether these capabilities are
  available in another configuration (no claim about licensing).
- Raw capacitances: the shared StretchSense plugin defines the OSC address
  `/v1/animation/capacitances/all` and the message "Enable Open SDK
  animation/slider/all in Hand Engine Settings", i.e. the raw stream belongs
  to StretchSense's Hand Engine product, not to XR Train's settings (which
  is why the user found no capacitance option). The glove's own
  configuration protobuf has `bleOutputFrequency`, `imuPollingFrequency`
  and `LowLatency` fields, not exposed in XR Train's UI.
- PlayerPrefs hold only sensitivities, paired gloves and window state: no
  per-hand smoothing or filter setting exists for the user to check.
- Batch runner for the 21 professor frames: `scripts/leap/record_prof_frames.py
  --reference "..\xr trainer\xr trainer poses"` (hand taken from each
  frame's file, resumable, `--dry-run`).

ChatGPT review of improvements 1 to 3 (2026-09-22) and the follow-up
(merged the same day, 339 tests)
- Review: keep the template fit but only as one fixed per-session
  calibration from well-observed open-palm frames (never per pose or
  frame); the 12 % effect is "pose-dependent scale variation in the
  tracker's reconstructed skeleton", not the hand changing size, and the
  fitted pinch agreement is not an independent validation of the camera.
  Lag: a per-hand constant shift is a sound first-order correction, applied
  only to pairing, only when several clips agree; it cannot correct creep.
  The disagreement gate compared curls on two different scales, so the
  77 -> 94 % thumb-use gain was partly a gate-calibration effect: compare
  normalised flexion fractions instead. Report all three as pipeline
  improvements, not as proof of accuracy.
- Follow-up: the thumb vote now thresholds the median |flexion fraction
  difference| at 0.20, each sensor / hand / finger on its own learned open
  and flexed endpoints (glove open = rail, camera open = median of
  open-palm-like frames, flexed = 2nd percentile; span guard 0.4 / 0.3,
  never triggered on real data). With that, fitted and unfitted runs ask
  the same question: thumb camera use day 1 77.3 % unfitted vs 77.5 %
  fitted (old gate: 77.4 -> 93.9), day 2 80.8 vs 81.7 (old: 82.8 -> 90.9);
  the one real residual is day-2 right thumbs up, whose disagreement sits
  on the gate (median 0.22 vs 0.21). Headline fused classifier unchanged
  and now identical fitted or not: 57/59 (day 1), 53/59 (day 2) with the
  profile; the ordinary run costs one take (55, 51). The fit's genuine
  effects remain: pinch index offset +0.14 -> 0.00, open-palm palm fit
  6.5 -> 4.3 mm.
- Lag is applied only with at least two trustworthy clips agreeing within
  60 ms MAD. Template fit refuses a hand with fewer than 200 open-palm
  frames and fuses it on the raw template.

Fusion improvements 1 to 3 (merged 2026-09-21, 328 tests): profile by
default, glove lag correction, template fit
- `profiles/default.json` (this operator's glove pair) is applied
  automatically; `--profile none` turns it off; the report always shows the
  ordinary and the profile-masked fusion side by side.
- Lag: the coached recorder now saves each take's 1.5 s open-palm -> pose
  transition as `<take>.settle.jsonl` (take files stay hold-only).
  `fuse_poses.py --glove-lag auto` estimates the glove's lag per hand from
  those clips (index-curl cross-correlation; needs camera motion, r > 0.8,
  consistent estimates) and shifts the glove's pairing stamps back by it;
  on held poses it reports "not measurable" and applies nothing. A forced
  0.46 s shift on day 1 moves glove-owned curl medians by at most 0.012 but
  camera-owned angle rows by up to 6 deg on single held takes (re-pairing),
  so the shift is only ever applied from a measurement.
- Template fit: bone and palm lengths are measured per hand from the
  session's camera frames (open palm preferred), the glove's template is
  rescaled to them keeping every rotation (`cam_hand/template_fit.py`,
  `--fit-template auto`, measurement saved as `<input>/template_<hand>.json`).
  Results, day 1 / day 2: fused pinch index curl minus camera +0.141 /
  +0.139 -> -0.004 / -0.007; thumb camera use 77.4 -> 93.9 % / 82.8 ->
  90.9 % (the glove's curls now sit on the camera's scale, so the
  disagreement vote passes); spread ring +3.6 / +6.2 points; override
  activation identical; fused classifier 55 -> 57/59 and 52 -> 53/59
  (glove and camera unchanged); palm fit on open-palm takes 6.5 / 6.7 ->
  4.3 / 4.3 mm. `curl_gate` is now the same fraction of each finger's
  learned open value (identical without a fit).
- Finding about the camera: LeapC returns every segment of the CLOSED
  gloved hand about 12 % shorter than the open hand, uniformly, both hands,
  both sessions. The camera's hand shape is consistent, its absolute size
  is not; so the all-frame palm RMSE rises 3.0 / 3.3 -> 7.7 / 7.0 mm after
  fitting to the open hand. Nothing downstream uses it (alignment is by
  palm basis and wrist; every reported quantity is a ratio or an angle).
  Scale factors on day 2: left thumb 1.18, index 0.95, middle 1.00, ring
  0.96, pinky 0.96; right thumb 1.17, index 0.91, middle 0.90, ring 0.87,
  pinky 0.88 (fitted over template).

Hardened diagnostics, first three runs (2026-09-20, 21:22 to 21:25;
`results/diagnostics/`)
- Tools merged (288 tests): `scripts/leap/hold_test.py` (camera must confirm
  the requested pose for 1.5 s before the clock starts; open-palm baseline
  saved), `scripts/leap/finger_sweep.py` (paced 5 s bend / 5 s straighten
  cycles, coverage check), both logging the glove packet rate per second;
  `fuse_poses.py --exclude`, `--profile profiles/reality_glove_nk_2026-09.json`
  (two-number report: day 2 thumb camera use 82.8 -> 99.6 %, fused 52 ->
  53/59), rail override per hand.
- Right fist hold, both gloves connected, glove 60 Hz: creep only +0.06 to
  +0.09 in 50 s (day 1: +0.26 on the index); the camera itself moved -0.04
  to -0.13 (fist tightened). Lag on the closing transition: right glove
  about 455 ms behind the camera (day 1: 485 ms).
- Right fist hold, only the right glove connected: the camera saw the fist
  loosen by about 0.3 on every finger over the minute while the glove moved
  0.02 to 0.11: the glove under-followed a slow relaxation. Stream 60 Hz in
  both runs, so the day-2 46 Hz problem did not recur and the one-glove test
  could not discriminate; no lost packets in any run.
- Right ring sweep, paced: the camera did not follow an isolated ring bend
  (camera ring 1.68 -> 1.42 -> 1.68 while the glove swung 1.97 -> 0.87),
  correlation 0.30 (the unpaced sweep at 20:04, where other fingers probably
  co-flexed, reached camera 0.82 with correlation 0.81). No transfer curve or
  saturation point can be read from it.

ChatGPT on the three runs (2026-09-20), adopted
- Right-ring rail override stays off: "stable camera geometry while the
  glove is wrong" was not met; the reliability mask stands on the static
  evidence. Run 2 shows a different temporal response, not "sticking" as a
  settled mechanism.
- Whole-hand open -> fist -> open paced sweeps are the primary curl
  diagnostic (the camera tracks that motion well and gives all four fingers
  a reference at once); isolated sweeps only for index and thumb; the sweep
  tool must flag a run when the camera's own range is too small; IR stills
  are qualitative evidence only. (Being built: `--finger all`, camera-range
  guard.)
- Wording: "glove curl shows time-dependent lag/hysteresis relative to the
  camera in both directions, with session-dependent magnitude (0.05 to 0.3
  of the curl metric per minute)". Repeat the same hold at the start and end
  of a session to test wear-time dependence.
- Lag 455 to 485 ms on the right at a clean 60 Hz makes packet-rate limits
  unlikely; check XR Trainer's per-hand smoothing / filtering /
  stabilisation / prediction / latency settings and confirm left and right
  are identical.

Day-2 repeat (2026-09-20, 19:51 to 20:02, identical protocol, both gloves
recalibrated first; `recordings/sync_day2`, `sync_day2_clean` without the
mislabelled take)
- 60/60 takes, 13 rejected attempts kept (hand lost; 9 left, 4 right). Pose
  check 42 ok, 18 warn, 0 mismatch. The per-take IR still caught
  pinch_right_take1 as a PEACE hand (operator had not changed pose); pinch
  is warn-only by design, so it was excluded by hand (59 takes).
- Glove stream worse: median 46 Hz per take (day 1: 60), worst gap 142 ms,
  4 takes with gaps over 100 ms (day 1: none). Same laptop.
- LEFT glove repeats day 1 within 0.1 on every finger and pose (one
  operator half-curl aside). RIGHT glove worse despite recalibration: in
  index point middle 1.96, ring 1.65, pinky 1.69 (camera 0.80 to 0.84);
  in take order it relaxes to the open rail within the five takes
  (middle 1.70 1.74 1.96 1.96 1.97, camera flat at 0.77 to 0.87). Fist and
  open palm fine on both hands. Left creep again (peace pinky 0.95 to 1.28
  over five takes, camera 0.80 to 0.94).
- Pinch: camera index 1.21 to 1.28, gap 0.05 to 0.25; glove on rail 9/9;
  override active 85 to 96 % of pinch frames (no take lost to the view
  gate this time, max 48 deg), fused index 1.34 to 1.39, 0 % elsewhere.
- Spread: glove constants again; camera open palm 21 / 17 deg (day 1
  24 / 20).
- Classifier: glove 40/59, camera 59/59, fused 52/59 (53 with
  `--unreliable right:middle,ring,pinky`); day 1 was 43 / 59 / 57. The
  fused misses are right-hand index point and peace takes where the glove's
  middle/ring/pinky read open.
- Diagnostics: only one of four holds and one of five sweeps were run. The
  right-fist hold is invalid (both sensors show an open palm for 60 s: the
  tool did not check the pose). Right ring sweep (45 Hz glove, 102 ms worst
  gap): glove vs camera correlation 0.81; binned by camera curl the glove
  reads 1.09 at camera 0.9 to 1.0, 1.50 at 1.0 to 1.1, 1.76 at 1.2 to 1.3,
  1.96 at 1.4 to 1.5 and its rail from 1.5 up: the right ring saturates to
  open once the finger is only about 40 % bent; hysteresis under 0.06.

ChatGPT review of day 2 (2026-09-20), adopted
- Wording: "XR Trainer's output for the right middle/ring/pinky drifts
  toward the open-palm rail under sustained mixed poses", not "channels
  lose flexion"; "all poses on the left glove" is too strong because pinch
  fails on the left too. The ring sweep is consistent with the same
  failure (compressed range then rail saturation) but does not prove the
  same physical/software cause.
- Reliability is explicit and per hand/finger: a saved profile (right:
  middle, ring, pinky unreliable), and the report shows ordinary fusion
  and profile-masked fusion side by side.
- Rail override per hand and per finger: right ring probably now; middle
  and pinky after their own sweeps. Minimum evidence per finger: repeated
  static failure on 2+ sessions, stable camera geometry while the glove is
  wrong, that finger's own sweep showing rail saturation.
- Rate drop strengthens "variable transport/software pipeline", still does
  not prove Bluetooth. Cheapest test: the same transition test with only
  ONE glove connected; if rate and gaps recover, contention. Next: external
  Bluetooth dongle away from USB 3, 5 GHz Wi-Fi or 2.4 GHz off.
- Diagnostic tools: hold test must require the camera to confirm the
  requested pose for 1 to 2 s before the clock starts and save the open-palm
  baseline; sweep must be paced (5 s bend / 5 s straighten cycles) with
  coverage required across intermediate curl bins; both log packet
  counters and rates continuously.
- Headline statements for the professor (as reworded): spread fixed in XR
  Trainer, pose-dependent on the camera; pinch, both days, index on its
  open value while the camera shows flexion and a small gap; curl:
  open palm and fist repeatable on both gloves, left glove generally
  repeatable for the non-pinch coached poses, right middle/ring/pinky
  reproducibly under-report and drift in mixed poses across two days
  despite recalibration; creep observed (strongest quantified evidence is
  day 1, so "observed" not "replicated"); lag 100 to 500 ms measured in one
  transition experiment, delivery rate 60 to 46 Hz between days. Do not
  claim a defective glove, a Bluetooth root cause, or a general XR Trainer
  dead zone.

Rail-disagreement override, trust-based thumb vote, report changes (merged
2026-09-18, 253 tests)
- `RailOverrideParams`: rail learned per hand/finger as the mode of the
  glove curl (tol 0.005; recovered 1.974/1.975 index, 2.072/2.075 middle,
  1.970/1.973 ring, 1.707/1.710 pinky, 1.426 thumb); camera open reference
  a constant per finger (index 1.75) minus margin 0.25; enter after 10
  qualifying frames, leave after 5; index only by default. The camera's
  bone directions are put on the glove's bone lengths from the knuckle out.
- On `sync_day1`: active on 77.9 % of pinch frames, 0.00 % on fist, index
  point, peace, thumbs up, 0.77 % on open palm (one real movement). Fused
  pinch index 1.98 -> 1.40 to 1.56 (camera 1.17 to 1.39; the +0.13 to +0.18
  is the template's longer index, agreement is within 0.02 as a fraction of
  each sensor's open value); fused thumb-index gap 0.94 -> 0.36 (camera
  0.28). Pinch L1 and R1 stay glove-only: viewing angle 52 deg fails the
  50 deg gate, and that bimodality costs the classifier one take (56 ->
  55/59 at the default gate, 57/59 at 55 deg; default left at 50).
- Thumb vote counts only fingers that are measuring: excluded when in the
  override, listed unreliable (`--unreliable right:ring,pinky`), or on the
  rail WHILE the camera reads that finger flexed (a railed finger the camera
  agrees with still votes; excluding every railed finger had dropped peace
  right from 33 % to 7 %). Fewer than two usable fingers = no glove veto.
  Thumb camera use overall 77.1 -> 77.4 % (rule alone) -> 85.5 % with
  `--unreliable`; thumbs up right 14 -> 47 %, peace right 33 -> 99 %;
  classifier 57/59 with `--unreliable`. The vote also surfaces the right
  MIDDLE finger (1.57 vs camera 0.93 in index point) as suspect.
- Report: "spread thumb-index" (base bone angle) dropped; thumb direction in
  the palm frame added as descriptive only; rejected attempts named with
  the hand.

Right-glove recalibration check and pose-hold drift (2026-09-18, 19:01 to
19:11; `recordings/right_recal_check`, `..\xr trainer\hold_drift_*.csv`)
- Reseat + fresh Basic calibration, then fist / peace / thumbs up / index
  point x 3 (12 takes, none rejected, pose check 6 ok 6 warn). Right ring:
  fist 1.03 -> 0.82, peace 1.51 -> 1.00 (fixed). Right pinky: fist 1.28 ->
  0.94, peace 1.62 -> 1.42, thumbs up 1.59 -> 1.45 (half fixed; still reads
  half open in mixed poses while the camera reads 0.8). Defensible
  statement for the professor, as an observation, not a defect.
- Creep: glove curl rises while a pose is held and the camera stays flat.
  Day-1 take order showed it (left peace pinky 0.92 -> 1.36 over five takes,
  camera 0.79 to 0.83). 60 s fist holds (`measure_hold_drift.py`): right
  index +0.26, ring +0.17, middle +0.12, pinky +0.11; left index +0.20,
  others +0.06 to +0.07; camera within +/-0.03 (right) and +/-0.06 (left).
  +0.26 is about 20 % of the glove's open-to-fist range in one minute.
- Lag by cross-correlation on the close/open transitions: right glove trails
  the camera by about 485 ms, left by about 100 ms (camera frames ~10 ms
  old). Yesterday the left was the slow one and dropped out for 3.1 s; today
  neither hand had a gap over 25 ms in 60 s. So the slow hand changes
  between days: not a fixed per-glove filter.

ChatGPT on the recalibration check and creep (2026-09-18), adopted
- Creep is consistent with viscoelastic creep/hysteresis in soft stretch
  sensing but cannot yet be located in the capacitive element; fabric/fit
  and XR Trainer filtering/calibration may contribute. 485 ms is too large
  for a BLE connection interval alone.
- Protocol: recalibrate at the start of every session; keep the open-palm
  reset and 5 s takes; standardise the analysis window (early 2 to 3 s
  after settle) for static-pose metrics. Do not invent a universal drift
  threshold: learn it from camera-stable hold trials and mark a glove DOF
  low-confidence when its within-hold drift exceeds that hand/finger's
  normal 95th percentile while the camera is stable. (To do: `--window` in
  fuse_poses.py; drift-confidence in fusion after the day-2 holds.)
- Best discriminator for lag and creep: record raw glove capacitance next
  to the solved kinematics and cross-correlate both against the camera.
  StretchSense software can expose `animation/capacitances/all` in some
  configurations; check whether XR Trainer can stream it (the OSC sniff of
  2026-09-17 did not see it with the default settings).
- Day 2: identical primary protocol, then diagnostics: 60 s fist and peace
  holds per hand, a repeated close/open transition block (with capacitance
  if available), the slow per-finger sweeps.

ChatGPT review of day 1 (2026-09-18), adopted
- Do not conclude the right glove hardware is worse: reseat the right
  glove, fresh Basic calibration, repeat fist / peace / thumbs up / index
  point three times; only a persistent under-read justifies "less reliable
  on those DOFs".
- Thumb gate compares only against trustworthy glove fingers (not on rail,
  not in the override, not listed unreliable), at least two usable fingers;
  with fewer, the camera's own gates decide.
- Drop the "spread thumb-index" row; keep the tip gap; add thumb direction
  in the palm frame as a descriptive row only.
- Report fused 95 % vs camera 100 % exactly as it is; classifier stays
  secondary.
- No documented left-hand tracking disadvantage: for day 2 close the left
  fist slowly after acquisition, keep knuckles visible, check the IR view
  for glare first. Day 2 = identical protocol, then a separate diagnostic
  block (slow flex/extend sweep per finger per hand, 2 to 3 cycles, right
  ring and pinky first; optionally three half-fists per hand), kept out of
  the primary dataset.

Third Path A session, coached recorder (2026-09-17, 15:34 to 15:41, both
hands one at a time, 6 poses x 3 takes each; copy in
`recordings/sync_coached_20260917`)
- Capture worked: 36/36 takes, coverage 100 % on 35 and 93 % on one, one
  hand id per take, heights 18.7 to 27.6 cm, 868/869 glove frames paired
  within 50 ms, palm fit 3.2 mm median.
- Labels did not: in 9 takes BOTH sensors show a different pose than the
  label (fist_left_1, fist_right_1/2 = open palm; index_point_left_1/2/3 and
  thumbs_up_left_1/2 = fist; peace_left_1 = thumbs up). Cause: the camera
  window showed the phase but never the pose name, and the operator was
  told to watch the window. The classifier numbers from this session
  (61 / 69 / 72 %) are on polluted labels and are not to be quoted.
- Pinch: the camera sees a pinch in 6/6 takes (index curl 1.20 to 1.30,
  thumb-index gap 0.10 to 0.35); the glove reports a bit-exact open palm in
  5/6, on both hands.
- Single-finger sweep (left index): only two fast bends, so sparse, but the
  glove left its open value as soon as the camera saw flexion, read about
  1.38 at 50 to 60 % bend and 0.66 at full bend. So there is NO general
  near-extension dead zone; the loss is specific to the pinch shape. The
  glove also returned slowly (1.87 to 1.90 for about 5 s, then snapped to
  1.9737). The left glove's stream stopped for 3.1 s during the sweep,
  which matches the on-screen lag complaint.
- XR Trainer's OSC output (sniffed): `/v1/animation/kinematic/all`,
  `/v1/orientation/all`, `/v1/controller_input/all` (12 button/axis fields,
  all zero at rest), `/v1/calibration/gesture/state`,
  `/v1/calibration/articulation/state` = `'Basic', 4, 100.0` on both hands,
  plus tracker config. Device label "Reality Glove". No raw stretch-sensor
  channel is exposed. Open lead: the articulation calibration level is
  "Basic"; a higher level, if XR Trainer has one, may represent the pinch.
  `..\xr trainer\probe_pinch_inputs.py --hand left` logs skeleton and
  controller values through open / pinch / open / fist to see whether
  anything XR Trainer sends carries the pinch.

ChatGPT review of the third session (2026-09-17), all adopted
- The pose check uses the two sensors under evaluation as the arbiter, so
  it is QC, not ground truth: rejected attempts are kept on disk under
  `rejected/` with the reason, the count is reported, and each take gets an
  independent IR still from the camera window for visual confirmation.
- Pinch acceptance by the camera's thumb-index gap would bias any camera
  pinch result, so for pinch the check only warns, never rejects.
- Change B renamed from "censored rail" to "rail-disagreement override":
  glove on its open-palm rail AND trusted camera shows sustained flexion
  for N frames, hysteresis, index finger in pinch-like shapes only until
  there is evidence for other fingers. No claim about a dead-zone width.
- Glove recorded at full rate in paired takes (was 5 Hz, 24 frames per
  take), downsampled offline if rates must be equal; stream gaps go in the
  take's meta.json.
- Sample size: 3 takes is pilot evidence. Minimum for a within-operator
  conclusion: 5 clean takes x 6 poses x 2 hands on 2 separate days; the
  take, not the frame, is the independent sample.

Second Path A session and what it changed (2026-09-17, 13:52, left hand
only, 6 poses x 3 takes)
- Glove side complete. Camera side: open_palm 3/3 with one continuous hand
  id; fist 34 % / 0 / 66 %; index_point almost nothing, once tracked for 3 s
  but labelled a RIGHT hand; thumbs_up never; pinch never as left; peace 1/3.
  The tracked id changed on almost every take and was often acquired late.
- Reading: the tracker follows an open hand into a pose but cannot acquire a
  gloved hand that is already closed, and when it re-acquires from a closed
  pose it sometimes picks the wrong chirality (a mirrored skeleton, not just
  a wrong label). The hand also sat at 131 to 154 mm; good takes were 160 to
  210 mm. The idle other hand was picked up 20 cm to the side.
- Fix, merged (179 then 183 tests): the recorder coaches each take as
  ACQUIRE (open palm, expected hand, in band, palm facing, centred, tracked
  500 ms) then "NOW: <pose>" with a settle, and records only if the same
  hand_id survived; otherwise both files are discarded and the take retried.
  Frames of the other chirality are never written. `<take>.meta.json` holds
  coverage, attempts, height and viewing angle. The gate uses the same
  module for a timed per-pose schedule and a pose-by-pose verdict against
  the bare hand.
- Live camera window (`scripts/leap/camera_view.py`), opened by every
  real-camera script and usable on its own: IR image, fitted skeleton,
  height against the band, palm facing, wrong-hand warning, the script's
  instruction as a caption. The 3D-to-pixel mapping
  (`LeapRectilinearToPixel` with slopes -(x - 32)/y and z/y for the left
  lens) was checked against the saved gate stills.

Gated fusion (ChatGPT-reviewed design, merged, 159 tests at the time)
- Spread is read from the proximal bone. A finger's spread comes from the
  camera only when the glove says it is not curled and the palm faces the
  lens within 50 degrees; the thumb comes from the camera only when the
  camera also agrees with the glove about the other four fingers (median
  curl disagreement below 0.35). Frame gates: visible 0.3 s, no recent id
  change, central field. Nothing is dropped; a rejected frame is the glove.
- On the first session: thumbs_up uses the camera for 0 % of every DOF
  (edge-on at 70 to 78 degrees, curl disagreement 0.88), as intended. The
  report's headline is now per DOF with camera-use rates and rejection
  reasons; the classifier is secondary and leave-one-take-out.
- Open issue, the glove's dead zone: during pinch the glove's five curls are
  bit-identical to its open palm, on both hands, in July and now, while the
  camera sees the index flex from 1.76 to 1.27. Since the glove owns curl,
  a fused pinch stays open. Reviewed extension, not yet built: treat a
  glove finger sitting on its open-palm rail as possibly censored, and take
  that finger's curl from the camera only when the camera passes its gates
  and shows flexion beyond a margin for several frames, with hysteresis,
  index finger first. Before telling the professor, measure it:
  `..\xr trainer\measure_dead_zone.py --hand left --finger index` sweeps one
  finger slowly and reports the dead-zone width, its hysteresis and the
  glove's lag against the camera (validated on a synthetic 30 % / 120 ms
  sweep: recovered 30 % and 115 ms).

Path A review round (ChatGPT, 2026-09-17; all fixed, 146 tests)
- Pairing clock. Frames were stamped when written, not when captured, so a
  drained burst of camera frames shared one time. Both sensors now carry
  `capture_time`: the camera's is computed in the LeapC callback as
  `time.time()` minus the frame's age on the Leap clock, the glove's is
  stamped when the OSC packet is enqueued. `fuse_poses.py` pairs on it when
  every frame has it and says which clock it used. Measured on a mock take:
  partners chosen on `wall_time` were 10.7 ms apart on average and 50.7 ms
  at worst in real capture time; on `capture_time`, 2.8 ms and 4.9 ms.
- The countdown beep blocked for 250 ms after the recorders had started, so
  every take opened with stale frames. Order is now beep, drop the backlog,
  open the files, in all four guided scripts.
- A take now needs the same hand on both sensors, not just frames on both.
- Alignment for metric camera data is palm-basis rotation plus wrist
  translation, scale 1, instead of a 5-point rigid fit. With only the palm
  proportions changed (the XR Trainer template versus a real hand), the
  palm basis tilts exactly 0 degrees while the point fit tilts 0.35 to 0.70
  degrees, straight into the spread the camera is there to supply. The
  point-fit RMSE is still reported, as a diagnostic, never a threshold.
- Professor-frame medoid is chosen after rigid palm alignment to the take's
  mean, and the original frame is exported. Over a 10 degree drift the raw
  wrist-centred distances of one pose spanned 48x; after alignment, 1.0x.
- A take name present under both `cam/` and `leap/` is now an error unless
  `--camera cam` or `--camera leap` says which.

Phase 2 tooling (2026-09-16, still the same evening, camera attached, no
second person available to hold a hand)
- Written: `src/leap_hand/images.py` (image policy, LeapC buffer -> numpy,
  `ImageSampler`, `HandTrail`, snapshot writing), `src/leap_hand/gate.py`
  (the thresholds, the verdict and the report text),
  `scripts/leap/{ir_snapshot.py, gate.py, convention_check.py}`, the
  `check_setup.py` split, and 26 tests. 107 pass.
- IR images confirmed on hardware, empty scene: 384 x 384, `bpp` 1, one
  plane per eye, mean pixel 5.3 (left) / 5.6 (right) in a dark room, and the
  PNGs show the room, so the policy, the pointer arithmetic and the copy are
  all right. `PolicyFlag.Images` comes back active from
  `set_policy_flags`.
- Raw `.lmt` capture confirmed on hardware for the first time: a 3 s gate run
  with an empty scene wrote 322 tracking events, 0 hands, and the file reads
  back through `leap.Recording`. That closes the last Phase 0 item.
- `check_setup.py` now splits the old "live tracking" line into "device
  streaming" (FAIL if no tracking events) and "hand seen" (WARN, exit 0),
  because an empty room is not a broken machine and this machine is usually
  empty.
- Still open, and it needs a second person: every number in the gate itself.
  Nothing has been measured with a hand — bare or gloved — so Path A vs B is
  undecided. The tooling was exercised with `--mock` end to end and with the
  device on an empty scene.

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
