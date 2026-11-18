# Gates: identify the glove stream and prepare a paired pilot

OWNS: scripts/glove_camera_pilot_check.py, tests/test_glove_camera_pilot_check.py, docs/glove_camera_pilot.md, launchers/glove_camera_pilot.ps1, GATES_glove_camera_pilot.md

Scope: identify the Reality Glove labels from local XR Train logs; distinguish solved skeleton coordinates from physical sensor channels; audit existing paired left-hand recordings; rehearse the existing simultaneous recorder; check live availability without substituting mock data for hardware evidence. Outputs go to results/glove_camera_pilot_20261025.

- [x] P1: the audit refuses incompatible inputs and reports known angle ranges and sensor disagreement correctly
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_glove_camera_pilot_check.py -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=cd89101887e8bd4b53f0ca77a88208b0e4f171bd87473b17d932b5eeb5b62c5d; exit=0; EXPECT=matched; output-sha256=db099e3469ad3ea61657fb24d5c76c7e02a471897687faacbf5b4b05df8ef640; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [x] P2: the existing paired real recordings are audited with source hashes, angle ranges, timing qualifications and per-joint disagreement
  CHECK: .venv\Scripts\python.exe scripts\glove_camera_pilot_check.py audit --session recordings\sync_day2 --hand left --out results\glove_camera_pilot_20261025\historical_gate
  EXPECT: PILOT AUDIT OK: pairs=3
  EVIDENCE: automatic-evidence=v1; definition-sha256=aaa2a8de75733fdb9689f6063424a01446786dc8729e21e655ac4fd67c487679; exit=0; EXPECT=matched; output-sha256=723f0d0dba50cc86d3a9af0c6f0251f4a93fd5de9f044fa2029c60285f408a58; output-bytes=450; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [x] P3: the simultaneous recorder completes a mock open-palm hold through its normal capture path
  CHECK: .venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --mock-glove --mock-leap --poses open_palm --takes 1 --duration 4 --prep 0 --settle 0.2 --acquire-timeout 8 --retries 0 --no-view --out-dir results\glove_camera_pilot_20261025\mock_capture_gate
  EXPECT: /Session summary \(left hand\): 1\/1 takes complete/
  EVIDENCE: automatic-evidence=v1; definition-sha256=a99258fc8dc97e29ddca4c9cae81b1febe7b1d0ff4552b69ca09b83705a5de5d; exit=0; EXPECT=matched; output-sha256=8a9ac42afee9e10bb0ae59de8853e0039d378f72eade4822b9d1fe59f451d41a; output-bytes=3715; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [x] P4: the captured mock pair is readable and compared as synthetic software evidence
  CHECK: .venv\Scripts\python.exe scripts\glove_camera_pilot_check.py audit --session results\glove_camera_pilot_20261025\mock_capture_gate --hand left --poses open_palm --data-origin synthetic --out results\glove_camera_pilot_20261025\mock_audit_gate
  EXPECT: PILOT AUDIT OK: pairs=1 origin=synthetic
  EVIDENCE: automatic-evidence=v1; definition-sha256=9db85412d185f762303ca8ec5823befaf3aeb98d70cb341c3a8e3d2c108c9607; exit=0; EXPECT=matched; output-sha256=6e10555e86c504bd5869279d300471ef7729a4773ddf668f1d57fcaedc347ea0; output-bytes=254; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [ ] P5: a new live simultaneous session is recorded with the connected gloves, connected Ultraleap and a participant, then compared

ABANDON: P5 the live probe received no OSC packets and the SDK saw no connected camera. The operator chose to prepare the procedure and record later. Run launchers/glove_camera_pilot.ps1 when both devices and the operator are available. Historical recordings and mock rehearsal do not pass this gate.

- [x] P6: existing pairing, joint-angle and pilot-comparison tests still pass
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_pilot_compare.py tests/test_joint_frames.py tests/test_record_simultaneous_hints.py -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=3c41606e24c20fb565688766b5edcb765d1ce3aa8a464e606592c95fe005c53d; exit=0; EXPECT=matched; output-sha256=ea1501e38477b8b28cfce14624411856e16c6850709dc6a37cbd6fa0203af19c; output-bytes=183; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

