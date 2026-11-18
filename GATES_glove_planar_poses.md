# Gates: 50 single-plane poses for the XR glove, not yet in the database

OWNS: scripts/leap/glove_planar_poses.py, tests/test_glove_planar_poses.py, docs/glove_planar_poses_50.md, protocols/glove_planar_poses_50.csv, protocols/glove_planar_poses_50.json, scripts/leap/glove_planar_poses_pdf.py, scripts/leap/pose_catalogue_html.py (the pose-table heading match only), results/glove_planar_poses/**, README.md, GATES_glove_planar_poses.md

Scope: a list of 50 static hand poses to record with the StretchSense XR glove (gloves on, no camera: amended 13 Nov 2026 on the user's correction that the glove recording should not use the camera; the first version assumed the Ultraleap as a reference) for a colleague's test of whether newly recorded poses satisfy the compiled constraint tables. Every pose moves each digit only in its own flexion-extension plane (no finger spread, no wrist deviation, no thumb opposition out of the palm plane), because that is what the glove senses. No pose repeats a recorded one (the ten paired poses, the July seven, the 23 mimed grasps, the Set C steps) or one of the 100 camera-catalogue (Leap) poses of 7 Nov 2026. Amended 11 Nov 2026 at the user's request ("the glove recordings should be different poses"): the exclusion is by FINGER SHAPE, the four finger states alone whatever the thumb, spread or wrist, because the glove does not sense spread or the wrist; the first version excluded only exact signatures and left 22 glove poses with a Leap pose's finger shape (24 under the strict rule, counting the recorded pinch family), all replaced. No two glove poses share a finger shape. The catalogue's templates and envelope are reused so the ranges are consistent across both lists. Each pose names the constraint it tests, its reference ranges and what the glove cannot sense in it. The list is written as a CSV, a recorder pose list (JSON) and a note with the one recording command. Nothing is recorded; nothing is committed.

- [x] P1: the generator runs in the repo venv and writes the CSV, the JSON pose list and the note, reporting 50 poses
  CHECK: .venv\Scripts\python.exe scripts\leap\glove_planar_poses.py
  EXPECT: /GLOVE PLANAR POSES OK: poses=50 /
  EVIDENCE: automatic-evidence=v1; definition-sha256=d441b9db50dd540a9871008fe5b1e93a65f9d0323b00b8e79615c1ded1b1c4b9; exit=0; EXPECT=matched; output-sha256=68e25f201366bb35980ba89f4ff547170fe312ba7cdfc82ae20a6d88eb177bce; output-bytes=506; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P2: the CSV has exactly 50 poses with unique ids, names and signatures; every pose is planar (spread together, wrist neutral, thumb in a planar state, no hyperextension state); no signature matches a database pose or a camera-catalogue pose, and no pose's four-finger shape is the finger shape of any Leap-list pose or recorded pose (read from the Leap CSV and the database list, with positive controls), nor of another glove pose; every range is the catalogue template for its states and lies inside the literature envelope; every pose names its setup (gloves on, no camera), the constraint it tests and a glove note; the JSON pose list has the same 50 ids in the same order and is accepted by the recorder's protocol loader
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_glove_planar_poses.py -q -k "csv or json"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=e0e9803a035078f6fb994f793b7912c2938645515f0de00a7966155f619f4565; exit=0; EXPECT=matched; output-sha256=826764158ac15956041ac6720b68a4850265c81975cda82b91575eb02cb47fcc; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P3: `docs/glove_planar_poses_50.md` has sections on what this is, why one plane, how to read the ranges, what is excluded, the recording setup with one command, and the 50 poses; its table lists every id in the CSV once; it names the hardware setup and the six constraint sources
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_glove_planar_poses.py -q -k "document"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=94668f9c1d6fb6dabdb61af70f81c426278a2e6fa0678f1f4df4ab35dff75c9c; exit=0; EXPECT=matched; output-sha256=1af1d11392167e6bcb4e33561b6182052defb108531a9731e55715fa0bfba43c; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P4: the camera catalogue's own tests still pass (its module is imported, not changed)
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_pose_catalogue.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=7f6d220ba9dd19bffb9cbaeefacdc9a68f1d178c7ed55679258e44b342b0eb80; exit=0; EXPECT=matched; output-sha256=bca794c6321f1de3bb05303c06918364e88677ac1880652e53e1298207505917; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P5: nothing is committed: HEAD is still 95b9040
  CHECK: git log -1 --format=%h
  EXPECT: /^95b9040/
  EVIDENCE: automatic-evidence=v1; definition-sha256=0e45c053f9dc1820407a11848e098f42669cbfc5dd88e3f01db015f4bc5d60f2; exit=0; EXPECT=matched; output-sha256=314e0e43b315f489baacf140ed4a172182d39ac5702bc9fb6e1dd9b68bdac6f0; output-bytes=8; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P6: the 50 glove poses print as one PDF with a MyoHand render pair beside every pose (added 11 Nov 2026): results/glove_planar_poses/glove_planar_poses_50.pdf, built from the existing renders by the MyoSuite venv; every one of its pages was looked at on 11 Nov 2026 (11 pages, no row cut, 100 pictures)
  CHECK: .venv-myo\Scripts\python.exe scripts\leap\glove_planar_poses_pdf.py --no-render
  EXPECT: /GLOVE POSES PDF OK: poses=50 renders=100 pages=\d+/
  EVIDENCE: automatic-evidence=v1; definition-sha256=c3a7f64502da73536602e461698cb4a33a6474b3c7ab01dde5d159dfbd0d79d7; exit=0; EXPECT=matched; output-sha256=ddde94c3e8c6ab2b227fd28d3448418ad7da721acf1dac9f79512d376d67ed1d; output-bytes=306; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries
