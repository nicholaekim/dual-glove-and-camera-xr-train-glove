# Gates: both hands in one take (--both)

OWNS: scripts/leap/record_poses.py, tests/test_record_poses_protocol.py, scripts/leap/pose_catalogue_protocol.py, tests/test_pose_catalogue.py, protocols/pose_catalogue_100_both.json, GATES_both_hands.md

Scope: the grasp recorder's protocol mode gets `--both`. The coaching still follows the operator's hand (`--hand`); the other hand makes the same pose beside it and is graded from the same take's lines with the same gate (`summarise_take`: 90 percent tracked, a static interval, the tracker's label). A take is kept only when both hands pass; the other hand's result is written to the take's meta (`second_hand`) and its summary frame to a second keypoint file. The catalogue gets a both-hands protocol whose picture shows both palms. Checked in mock mode only; no camera is run; nothing is committed.

- [x] B1: with `--both` a mock take is kept with the other hand graded in its meta and a second keypoint file; without it the meta has no second hand; a take whose other hand fails the gate is rejected with a reason that names that hand
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_record_poses_protocol.py -q -k "both_hands"
  EXPECT: /4 passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=88c8092c1b5f85a30a30213ac68be2a2101c27bee6bed5e1b8c2b43628063bcb; exit=0; EXPECT=matched; output-sha256=3903701cea8e400f8a0378489cd243c33015b4f5f871c5541afae100da71e78c; output-bytes=116; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] B2: the recorder's existing protocol tests still pass
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_record_poses_protocol.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=f69165109eb97576dc525eea9a65a69cf2695446f1f0ea2faeb12a93eaf1ad69; exit=0; EXPECT=matched; output-sha256=d764889befe0d9d273563326d86c5d448629da6d341adb51f0dd78e3be636d30; output-bytes=194; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] B3: `protocols/pose_catalogue_100_both.json` loads in the recorder, lists the 100 catalogue poses once each and every one has its picture on disk
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_pose_catalogue.py -q -k "both"
  EXPECT: /1 passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=19660645be62d482acf8e0ba79c61090a93e1430ce0e694cbd72b49d2f5e7531; exit=0; EXPECT=matched; output-sha256=cf5b3a94efda221c1c91ec2f8e5fdc6243595f14e17154550bcb00390ef986df; output-bytes=114; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] B4: nothing is committed: HEAD is still 95b9040
  CHECK: git log -1 --format=%h
  EXPECT: /^95b9040/
  EVIDENCE: automatic-evidence=v1; definition-sha256=0e45c053f9dc1820407a11848e098f42669cbfc5dd88e3f01db015f4bc5d60f2; exit=0; EXPECT=matched; output-sha256=314e0e43b315f489baacf140ed4a172182d39ac5702bc9fb6e1dd9b68bdac6f0; output-bytes=8; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] B5: with two hands in view on purpose, OPEN HAND acquires only the hand the tracker labels as the operator's (the left palm 16 cm to the side at 22 cm up, as on the real camera on 2026-11-12, is acquired and the centred right hand is not), and the coaching never adopts a new id under the other label
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_record_poses_protocol.py -q -k "two_hands_in_view"
  EXPECT: /2 passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=6fc95f9eeb77fb1536494a9619753d6a12ded3358f1ec85590b1304233d73bcb; exit=0; EXPECT=matched; output-sha256=363fdd11d5854b8fa882ea132b2932527eb7fd81e04291c11abc8e3f469e8058; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries
