# Gates: the pose catalogue's two false resets (--finger-poses)

OWNS: scripts/leap/record_poses.py, src/leap_hand/transition.py, tests/test_record_poses_protocol.py, GATES_finger_poses.md

Scope: the coached recorder gets `--finger-poses`, for the pose catalogue. With it (and only with it; the grasp set's rules, `RECORDER_RULES` 2026-10-07, do not change): (1) READY calls a hand closing before the GO beep only when its pose changes (the whole hand's movement taken out, `pose_speeds`, 25 mm/s), not when the hand travels over the camera; (2) MAKE THE GRASP does not call "no close after the cue" when a fingertip moved 150 mm/s or faster against the palm, so a pose that bends one finger counts. Both thresholds were read from the 51 takes kept on 2026-11-12. The session and every take's meta say the flag was on; the resume command carries it. Checked in mock mode and on synthetic frames; no camera is run; nothing is committed.

- [x] P1: with `--finger-poses` READY does not call an open hand carried over the camera at 100 mm/s a close and still calls a closing hand one; a one-finger bend after the cue is a close and a still hand is not; a mock session records the flag and its thresholds, a session without it records nothing new, and the resume command carries the flag
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_record_poses_protocol.py -q -k "finger_poses"
  EXPECT: /3 passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=d937569b322a1f437752a32bebf62f3320c58d3955eafa9d6fd17ea74bda2c13; exit=0; EXPECT=matched; output-sha256=bdf3854264d302f997c9fbdd83380ba8d02088a6f18525ae340031ca811a0c64; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P2: the recorder's protocol tests still pass, the grasp-set rules untouched
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_record_poses_protocol.py -q
  EXPECT: /\d+ passed in/
  EVIDENCE: automatic-evidence=v1; definition-sha256=c320f619fac471c7cddedacaa3ede4692f439dbcb700bba60ec1dd2d2deecb1e; exit=0; EXPECT=matched; output-sha256=47b98932363460a32893a4d24e322d93b446e7e37ed8f4c0a0c39b17864cad32; output-bytes=194; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P3: the transition tests still pass
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_transition.py -q
  EXPECT: /\d+ passed in/
  EVIDENCE: automatic-evidence=v1; definition-sha256=eba994bcc8468aa29dad895459b2d4763c51f91dde8dec0b1621d36abf92eca5; exit=0; EXPECT=matched; output-sha256=1fe47afe4ec14bbb19b70cf2afd7135edc779afc8cb262fa0cc6950be4205a4c; output-bytes=102; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries

- [x] P4: nothing is committed: HEAD is still 95b9040
  CHECK: git log -1 --format=%h
  EXPECT: /^95b9040/
  EVIDENCE: automatic-evidence=v1; definition-sha256=0e45c053f9dc1820407a11848e098f42669cbfc5dd88e3f01db015f4bc5d60f2; exit=0; EXPECT=matched; output-sha256=314e0e43b315f489baacf140ed4a172182d39ac5702bc9fb6e1dd9b68bdac6f0; output-bytes=8; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=81ce755f3d81/39 entries
