# Gates: hand_on_object hint toggle for the recorders

OWNS: src/leap_hand/stream.py, src/leap_hand/mock.py, scripts/leap/record_poses.py, scripts/record_simultaneous.py, tests/test_leap_hand.py, tests/test_record_poses_protocol.py, tests/test_record_simultaneous_hints.py, docs/grasp_recording.md, docs/pilot_hand_on_object.md

Scope: a `--hint hand_on_object` option on both recorders that requests Ultraleap's hand-on-object tracking per device, refuses to record if the service's Hyperion license is absent, and writes the requested hint and its confirmation into every take's metadata; plus the three-condition pilot protocol.

- [x] H1: with a hint requested, the stream calls LeapSetDeviceHints once per device with the exact NULL-terminated list, after the device wait and before any frame is counted, and records hints_requested / hints_confirmed / license_hyperion
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_leap_hand.py -q -k hint
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=e18604daa941ce05f31a61d5f634776a2125680de57d5d8e7e90e25c7c5efc0d; exit=0; EXPECT=matched; output-sha256=badac79348936e9149058597e9308374c66267db879f4d22f23793754e5639f2; output-bytes=117; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=d640fd9d8f83/37 entries

- [x] H2: with no hint requested nothing is called and the recorded metadata says so; with a hint but no Hyperion license the stream raises LeapUnavailable naming the next step
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_leap_hand.py -q -k "hint and (none or license)"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=eeddab1fe6821abd3f44cba901399e6d99da96dd1a3a8f6beb8264b78ecc5380; exit=0; EXPECT=matched; output-sha256=ae9f54bf60d6b020fee01f62103c94a46077f1e54760dea1ac07698a0cb5f23f; output-bytes=116; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=d640fd9d8f83/37 entries

- [x] H3: record_poses.py --hint writes tracking_hints, hints_confirmed and license_hyperion into session.json and into every take entry (mock run)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_record_poses_protocol.py -q -k hint
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=eb971d184051d6a54473cf536d503ee51d4b7cd1c1cb2d0cc6f52096d1ad2046; exit=0; EXPECT=matched; output-sha256=9e873e83cc823ab67459bc1162cab98c75de52319f770a5da554f659d55687c5; output-bytes=116; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=d640fd9d8f83/37 entries

- [x] H4: record_simultaneous.py --hint writes the same three fields into every <take>.meta.json (mock glove and mock leap)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_record_simultaneous_hints.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=5433153bfb121243819369ed1f6aa40c107cf5814f1f387ab83b8c1568551795; exit=0; EXPECT=matched; output-sha256=ede7084bbaa436ff262f8da695001f7efe2c9030a20e7eed31e52b31611e1da2; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=d640fd9d8f83/37 entries

- [ ] H5: the whole test suite still passes
  CHECK: .venv\Scripts\python.exe -m pytest tests -q -x
  EXPECT: / passed/
  EVIDENCE: pending

- [x] H6: the live service on this machine reports the Hyperion flag (no hardware); the live dry run with the camera attached is step 2 of docs/pilot_hand_on_object.md and belongs to the operator
  EVIDENCE: 2026-10-16: probe_hyperion.py (session scratchpad) against the running service 6.2.0+2025.07.11, no device attached: LeapCheckLicenseFlag("Hyperion") result 0, enabled True; "SetClassifierThresholds" also True. 2026-10-19, camera attached (IR 170, desktop mode), the protocol's step 2 dry run: `record_simultaneous.py --camera leap --mock-glove --hand left --poses open_palm --takes 1 --hint hand_on_object` printed "hint: hand_on_object (confirmed by the service, Hyperion license present)", accepted the take on attempt 2, and wrote tracking_hints ["hand_on_object"], hints_confirmed true, license_hyperion true into recordings/pilot_hoo/dry_run/leap/open_palm_left_take1_20261019_225024.meta.json. The first camera attempt on 2026-10-17 had failed: on the multi-device-aware connection the plain LeapGetTrackingMode is never answered; fixed by per-device LeapSetTrackingModeEx / LeapGetTrackingModeEx on the open handle (stream.py), with tests.

- [x] H7: docs/pilot_hand_on_object.md exists, names the hardware setup on every step, and its three conditions, grasps, take counts and analysis match what the recorders write
  EVIDENCE: 2026-10-16: read in full after three review rounds (ChatGPT thread: six corrections, then five, verdict VERIFIED on their application). Steps 1, 2, 2b, 3-10 each open with the setup (no hardware / camera only, bare hand / no sensors / gloves on + camera). Conditions A, B1, C, B2, A-check; 45 + 3 recordings; --hint and --note options and the three metadata keys match the recorders' implementation; C marked exploratory with the Ultraleap Tracking Models page cited (HMD only; Hyperion download for Leap Motion Controller 2).
