# Gates: recording the glove poses (a set of 10 since 13 Nov 2026), with a screenshot of the XR Train application for every take

OWNS: src/xr_hand/app_screenshot.py, src/xr_hand/recorder.py (load_pose_list), scripts/glove/record_poses.py, tests/test_glove_record_poses.py, scripts/record_simultaneous.py, tests/test_app_screenshot.py, tests/test_record_simultaneous_app.py, launchers/record_glove_poses.cmd (and its copy on the Desktop), docs/glove_planar_poses_50.md (through scripts/leap/glove_planar_poses.py), scripts/leap/glove_planar_poses.py, scripts/leap/glove_planar_poses_pdf.py, protocols/glove_planar_poses_recording_set.json, protocols/glove_planar_poses_50.csv, tests/test_glove_planar_poses.py, results/glove_recording_gate/**, results/glove_planar_poses/**, GATES_glove_recording.md

AMENDED 13 Nov 2026, the user's correction: "why does the xr train glove use the cmare? it shouldnt". The glove recording is GLOVE ONLY: gloves on, no camera. The first version of this work recorded through `scripts/record_simultaneous.py` (glove and Ultraleap together) on the assistant's own assumption that the camera was wanted as a reference; that was wrong, because the Leap poses are one set and the glove poses are a separate, different set. The launcher now runs the glove-only recorder `scripts/glove/record_poses.py`, which opens no camera, and the note, the pose lists and the sheet say gloves on, no camera. The screenshot and pose-list flags stay on the simultaneous recorder too (gates R2, R5), unused by this launcher. Gates R9 to R11 are the glove-only recorder; R4 is now its command. Where the scope below says camera, read this paragraph instead.

AMENDED AGAIN 13 Nov 2026, the user: "ok lets only do 10 poses for the glove." The recording set is now 10 of the 50 catalogue poses (ps01, pd01, pf07, pr07, pc01, pc05, pm01, pt01, pt02, pt03): every group, every finger state and the three thumb-joint tests are still in it. Because the count has changed twice, the count is out of the file names: the pose list is `protocols/glove_planar_poses_recording_set.json`, the launcher `launchers/record_glove_poses.cmd`, the sheet `results/glove_planar_poses/glove_planar_poses_recording_set.pdf`, the output folder `recordings/glove_planar_left`; the 25-pose files of earlier today are removed (nothing was recorded with them). Where the scope and gates below say 25, read 10 and these names.

Scope: one command that starts the recording of the single-plane glove poses (left hand). Asked for on 13 Nov 2026, in this order: "whats the command to start the recordings", then a screenshot of the XR Train application for every pose, then "lets only do 25 poses". So the recording set is 25 of the 50 catalogue poses (every group, every finger state and the three thumb-joint tests kept; the hardest finger-independence shapes and the mirror images left out), written as `protocols/glove_planar_poses_25.json`, marked in the catalogue's CSV and note, and shown alone in `results/glove_planar_poses/glove_planar_poses_25.pdf`. And a screenshot of the StretchSense XR Train application's window saved for every take, taken at the midpoint of the take's hold (the moment the camera still is taken), so each recording has a picture of the hand model the application showed. The screenshot is taken off the recorder's frame loop, is named like the take, follows the take when it is kept or rejected, and the take's meta.json says where it is, when and how it was captured, or why there is none; a recording never fails because a screenshot could not be taken, and nothing is reported as captured that was not. The recorder also shows each pose's name and description from the pose list, because ids such as `ps01` say nothing to the operator. Without the new flags the recorder behaves exactly as before. Nothing is recorded by this work; nothing is committed.

- [x] R1: the capture module finds a top-level window by title, saves it as a PNG and reports the time, method and size; a missing or minimised window gives a reason and no file; an all-black capture is saved and flagged as black, never passed off as a picture
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_app_screenshot.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=980c766ae99d85d61d03189c158d70a864342bda6b51a80bf742dec756e0feb3; exit=0; EXPECT=matched; output-sha256=decf5bee725079c4f851367c224167e5848c559a7b34cca5ff196fc0a4fb6ce6; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R2: on mock sensors through main(), `--app-screenshot` saves one PNG per kept take under `xr_train/` named like the take, a rejected attempt's PNG moves to `rejected/xr_train/`, every meta.json carries the screenshot keys (path, capture time inside the commanded hold, method, window), a capture that fails leaves the take kept with the reason in its meta, and a run without the flag writes no folder and no key; `--pose-list` supplies the ids and shows each pose's label and shape, `--start-at` begins at a named pose, and both flags are refused where they cannot work
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_record_simultaneous_app.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=231aae4c1f63af760e2e825243267a2996739c28d280103beb56e12d32333e4f; exit=0; EXPECT=matched; output-sha256=768de9816610021192d3ceacea83f0e861cf3ada9e2db47870459fb2ec4c4970; output-bytes=102; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R3: with the real XR Train application running on this machine, one capture of its window is a PNG at the window's size that is not black (checked on 13 Nov 2026; the picture was also looked at)
  CHECK: .venv\Scripts\python.exe -m xr_hand.app_screenshot "XR Train" results\glove_recording_gate\xr_train_live.png
  EXPECT: /APP SCREENSHOT OK: .* black=no/
  EVIDENCE: automatic-evidence=v1; definition-sha256=ba2b50ace437411bce6b2f011a4bbbb4a708b566595da277fc24e262f48233e6; exit=0; EXPECT=matched; output-sha256=1d4b7b655b17eedd83726af5587c8f6e60488343253d91639283097ace30bb74; output-bytes=173; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R4: the launcher's own recorder command runs end to end on mock sensors with the real XR Train window: 2 poses, 1 take each, every kept take with its screenshot, and the session's last line counts them
  CHECK: .venv\Scripts\python.exe scripts\glove\record_poses.py --mock --hand left --pose-list protocols\glove_planar_poses_recording_set.json --start-at pt02 --takes 1 --first-take auto --duration 1 --prep 0 --hz 0 --out-dir results\glove_recording_gate\mock_glove_session --app-screenshot "XR Train"
  EXPECT: /APP SCREENSHOTS: 2 of 2 takes have one/
  EVIDENCE: automatic-evidence=v1; definition-sha256=13db5edb929539cef8534ce1bac6ff66ff30e5db3e3b733013f046433517d144; exit=0; EXPECT=matched; output-sha256=8610857184046668f7c9cf755e116798becc90b97d6adc71c6cab489d206a535; output-bytes=1742; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R5: the recorder's existing tests and the tests of the tools that drive it still pass
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_record_simultaneous_hints.py tests\test_pilot_run.py tests\test_camera_view.py tests\test_pose_check.py tests\test_glove_planar_poses.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=776e70c3d9eab010e66eb234bd641f0d28f5cf47a1f98a12db06997579a3e041; exit=0; EXPECT=matched; output-sha256=2335f8178c9a12817337c8ddffe72fe367faedfe0a507c9e11d03e2da2f0a0e1; output-bytes=265; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R7: the recording set is 10 poses of the 50, in catalogue order, with every group, every finger state and the thumb-joint tests; its pose list holds the same items as the 50's; the note lists the 10 and gives the launcher and its recorder command, which is the command in `launchers\record_glove_poses.cmd`
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_glove_planar_poses.py -q -k "recording_set or document"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=25302f55174171a119eedf95db57650cf27b426af450e477eb644e5ecda1ad17; exit=0; EXPECT=matched; output-sha256=81e7e375436768f4ad8782b6b07fc251578fab786625b6e08faf38ac8b05999f; output-bytes=114; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R8: the sheet to have open while recording, `results\glove_planar_poses\glove_planar_poses_recording_set.pdf`, builds with the 10 poses and a render pair for each (its pages were looked at on 13 Nov 2026 after the cut to 10: no row cut, 20 pictures, its own title)
  CHECK: .venv-myo\Scripts\python.exe scripts\leap\glove_planar_poses_pdf.py --no-render --recording-set
  EXPECT: /GLOVE POSES PDF OK: poses=10 renders=20 pages=\d+/
  EVIDENCE: automatic-evidence=v1; definition-sha256=e111502d25fd8cc320c57903d5eef21b27b6c292a23b13ff48bc2a1ee3647b1e; exit=0; EXPECT=matched; output-sha256=5da8bdb8e7c7241c9cce6ee75bdb3aa21126d88866bdcb42e177638842d4aac1; output-bytes=325; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R9: the glove-only recorder, on its mock glove: `--pose-list` supplies the poses and shows each label and shape, `--start-at` begins at a named pose, `--hand left` keeps only the left glove's frames, `--first-take auto` numbers after the takes on disk; `--app-screenshot` saves one PNG per take under `xr_train/` named like the take, captured inside the take's hold, and every take gets a `<take>.meta.json` with the pose, label, shape, hold times, frames, note and the screenshot keys; a failed capture leaves the take kept with the reason; a take with no frames keeps neither file nor picture; the recorder refuses to start when the window cannot be captured; without the new flags it writes what it always wrote
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_glove_record_poses.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=f649e7aea626447d72d76ea27c97b53ea998d9c89f0eddb76d09f9b149beb00e; exit=0; EXPECT=matched; output-sha256=7b6d3ce7f1b622c9aa7e5be8b2a0ed8ba7985bf3012a0e4c02f3d6850c7dc5e6; output-bytes=101; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R10: the glove recording uses no camera: the launcher runs `scripts\glove\record_poses.py` and names no camera flag, the recorder script imports nothing of the camera packages, and the note, both pose lists and the CSV say gloves on, no camera
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_glove_record_poses.py tests\test_glove_planar_poses.py -q -k "no_camera"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=e52cfe18f218eca8765987c15ebf63bb14c0c267f4ad0f4c51cad55e9393708f; exit=0; EXPECT=matched; output-sha256=e7371d0e1d3cd29730413134f9474238606c12fd54984b3333095adc27940cf4; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R11: the two sheets were rebuilt after the correction and say gloves on, no camera (the recording sheet was looked at again on 13 Nov 2026)
  CHECK: .venv-myo\Scripts\python.exe scripts\leap\glove_planar_poses_pdf.py --no-render
  EXPECT: /GLOVE POSES PDF OK: poses=50 renders=100 pages=\d+/
  EVIDENCE: automatic-evidence=v1; definition-sha256=c3a7f64502da73536602e461698cb4a33a6474b3c7ab01dde5d159dfbd0d79d7; exit=0; EXPECT=matched; output-sha256=ddde94c3e8c6ab2b227fd28d3448418ad7da721acf1dac9f79512d376d67ed1d; output-bytes=306; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] R6: nothing is committed: HEAD is still 95b9040
  CHECK: git log -1 --format=%h
  EXPECT: /^95b9040/
  EVIDENCE: automatic-evidence=v1; definition-sha256=0e45c053f9dc1820407a11848e098f42669cbfc5dd88e3f01db015f4bc5d60f2; exit=0; EXPECT=matched; output-sha256=314e0e43b315f489baacf140ed4a172182d39ac5702bc9fb6e1dd9b68bdac6f0; output-bytes=8; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries
