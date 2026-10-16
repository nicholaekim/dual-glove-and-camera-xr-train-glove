# Gates: hand_on_object hint toggle for the recorders

OWNS: src/leap_hand/stream.py, src/leap_hand/mock.py, scripts/leap/record_poses.py, scripts/record_simultaneous.py, tests/test_leap_hand.py, tests/test_record_poses_protocol.py, tests/test_record_simultaneous_hints.py, docs/grasp_recording.md, docs/pilot_hand_on_object.md

Scope: a `--hint hand_on_object` option on both recorders that requests Ultraleap's hand-on-object tracking per device, refuses to record if the service's Hyperion license is absent, and writes the requested hint and its confirmation into every take's metadata; plus the three-condition pilot protocol.

- [x] H1: with a hint requested, the stream calls LeapSetDeviceHints once per device with the exact NULL-terminated list, after the device wait and before any frame is counted, and records hints_requested / hints_confirmed / license_hyperion
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_leap_hand.py -q -k hint
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=e18604daa941ce05f31a61d5f634776a2125680de57d5d8e7e90e25c7c5efc0d; exit=0; EXPECT=matched; output-sha256=b9d89dc3c84d2d63ca38bef80ee311e5c303005ce3d31d9b4e4cff6c63a57858; output-bytes=117; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=b05fcc5e3f1f/36 entries

- [x] H2: with no hint requested nothing is called and the recorded metadata says so; with a hint but no Hyperion license the stream raises LeapUnavailable naming the next step
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_leap_hand.py -q -k "hint and (none or license)"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=eeddab1fe6821abd3f44cba901399e6d99da96dd1a3a8f6beb8264b78ecc5380; exit=0; EXPECT=matched; output-sha256=955ac09374a34b7246b7aba1bd0cdb9b73fc299dbcd46a45fb4931d017301a3f; output-bytes=116; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=b05fcc5e3f1f/36 entries

- [x] H3: record_poses.py --hint writes tracking_hints, hints_confirmed and license_hyperion into session.json and into every take entry (mock run)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_record_poses_protocol.py -q -k hint
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=eb971d184051d6a54473cf536d503ee51d4b7cd1c1cb2d0cc6f52096d1ad2046; exit=0; EXPECT=matched; output-sha256=6ed9ff307030368470b397d77a67f45747f01e0ba38aeb5ef2bb6f1a6e89d94a; output-bytes=116; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=b05fcc5e3f1f/36 entries

- [x] H4: record_simultaneous.py --hint writes the same three fields into every <take>.meta.json (mock glove and mock leap)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_record_simultaneous_hints.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=5433153bfb121243819369ed1f6aa40c107cf5814f1f387ab83b8c1568551795; exit=0; EXPECT=matched; output-sha256=4ce6c9af27473e135bee7cd62471ddcd361f664f4229ec95685716675a901065; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=b05fcc5e3f1f/36 entries

- [x] H5: the whole test suite still passes
  CHECK: .venv\Scripts\python.exe -m pytest tests -q -x
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=82d9daa730e6f68dcc71f03b8a43c11ff501ce9ad583b6a269b29ac693626b8b; exit=0; EXPECT=matched; output-sha256=f8bd23a120523480f7073c1197aca4217ba44ca8ea4bcfb08ba5a97ed9af1bb4; output-bytes=1331; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=b05fcc5e3f1f/36 entries

- [x] H6: the live service on this machine reports the Hyperion flag (no hardware); the live dry run with the camera attached is step 2 of docs/pilot_hand_on_object.md and belongs to the operator
  EVIDENCE: 2026-10-16: probe_hyperion.py (session scratchpad) against the running service 6.2.0+2025.07.11, no device attached: LeapCheckLicenseFlag("Hyperion") result 0, enabled True; "SetClassifierThresholds" also True. The camera-attached confirmation of hints_confirmed is deliberately the protocol's step 2, done by the operator before the gloves go on.

- [x] H7: docs/pilot_hand_on_object.md exists, names the hardware setup on every step, and its three conditions, grasps, take counts and analysis match what the recorders write
  EVIDENCE: 2026-10-16: read in full after three review rounds (ChatGPT thread: six corrections, then five, verdict VERIFIED on their application). Steps 1, 2, 2b, 3-10 each open with the setup (no hardware / camera only, bare hand / no sensors / gloves on + camera). Conditions A, B1, C, B2, A-check; 45 + 3 recordings; --hint and --note options and the three metadata keys match the recorders' implementation; C marked exploratory with the Ultraleap Tracking Models page cited (HMD only; Hyperion download for Leap Motion Controller 2).
