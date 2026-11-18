# Gates: 100 new poses for the Leap Motion with reference joint ranges

OWNS: scripts/leap/pose_catalogue.py, tests/test_pose_catalogue.py, docs/pose_catalogue_100.md, protocols/pose_catalogue_100.csv, README.md, GATES_pose_catalogue.md

Scope: a catalogue of 100 static hand poses that the Ultraleap IR 170 can record (camera only, bare hand) and that are not already in the database (the ten paired poses, the July seven, the 23 mimed grasps, the Set C sequence steps), each with an approximate reference range for every camera-given angle (finger MCP flexion and abduction, PIP, DIP, thumb MCP and IP, wrist flexion and deviation) derived from the constraint tables compiled on 7 Nov 2026 (Dragusanu 2025, Isaac 2021, Ingram 2008, Chen 2013 = Cobos 2009, Jia 2021, MyoHand). One generator script writes the CSV and the note; the test proves the count, the uniqueness, the exclusion of database poses and that every range lies inside the literature envelope. No hardware is run; nothing is committed. The whole suite is not rerun: no existing module changes.

- [x] C1: the generator runs in the repo venv and writes `protocols/pose_catalogue_100.csv` and `docs/pose_catalogue_100.md`, reporting 100 poses
  CHECK: .venv\Scripts\python.exe scripts\leap\pose_catalogue.py
  EXPECT: /POSE CATALOGUE OK: poses=100 /
  EVIDENCE: automatic-evidence=v1; definition-sha256=afb5835f8f7720f047ccbaaa795e20ad7aca4d76409a2bb2cd6be264b24b8a70; exit=0; EXPECT=matched; output-sha256=33a4be4de9c2f698e8644758052ea3faca2f23b8df6d5ac18eeda1764a60ab9b; output-bytes=317; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] C2: the CSV has exactly 100 poses with unique ids, names and digit-state signatures; no signature matches a database pose (ten paired poses, July seven, sequence steps, mimed grasp shapes); every range has min < max and lies inside the literature envelope; every pose's ranges are the template ranges for its digit states; every pose names its setup and a Leap visibility note
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_pose_catalogue.py -q -k "csv"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=f0a6808894d0c72c12d51fd89def4fa8411a40963d776922fd15eef9636bdbb5; exit=0; EXPECT=matched; output-sha256=8903cad19b7d43a92d429af02b5f32d41f4153f7571b89e36a59ff9b59565d21; output-bytes=114; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] C3: `docs/pose_catalogue_100.md` has sections on what it is, how to read the ranges (templates, signs, envelope and its sources), what is excluded, the recording setup, and the 100 poses; its table lists every id in the CSV once
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_pose_catalogue.py -q -k "document"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=ab7fa871ec36d6c99e6583bdd3238e5478188626d0377fcddae8ce7a94d0e7bc; exit=0; EXPECT=matched; output-sha256=3ddfd84b050c84f662657fd6d3bf8ddd248a1e906754b8dc6c8cba3352549c7d; output-bytes=114; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] C4: nothing is committed: HEAD is still 95b9040
  CHECK: git log -1 --format=%h
  EXPECT: /^95b9040/
  EVIDENCE: automatic-evidence=v1; definition-sha256=0e45c053f9dc1820407a11848e098f42669cbfc5dd88e3f01db015f4bc5d60f2; exit=0; EXPECT=matched; output-sha256=314e0e43b315f489baacf140ed4a172182d39ac5702bc9fb6e1dd9b68bdac6f0; output-bytes=8; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries
