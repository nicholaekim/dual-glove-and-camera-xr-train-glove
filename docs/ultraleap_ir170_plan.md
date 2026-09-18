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
