# Gates: repository housekeeping, 2026-10-25

Scope: remove Python bytecode caches, pytest cache and obsolete cam_hand distribution metadata. Preserve all tracked files, all uncommitted source files, recordings, research outputs, archive sources, SDKs and installed dependency code. No source-code deletion was justified by the reference and duplicate-file checks.

- [x] C1: cleanup targets were confined to this checkout and all pre-existing tracked and untracked source files retain their original hashes
  CHECK: powershell -NoProfile -File results\repo_cleanup_20261025\cleanup.ps1 -Mode Verify
  EXPECT: CLEANUP PRESERVATION OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=6c95abdeafc125bc70dde91192178a590d9e8ff96ca5fcfe7a60b9f35aebb389; exit=0; EXPECT=matched; output-sha256=858fb0e57ec8b19923f3645baf8c0e112e78ea078158eacacdc14ac406d1955f; output-bytes=83; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [x] C2: current editable installation supplies all three packages and dependencies remain consistent
  CHECK: .venv\Scripts\python.exe -B -m pip check
  EXPECT: No broken requirements found
  EVIDENCE: automatic-evidence=v1; definition-sha256=32766d1a4a987a5dd9c3755344bb3ed5502f3439a76fe7c4bb8983d27a79ca0f; exit=0; EXPECT=matched; output-sha256=672aec65a45bed50e70229c70934a7eebf0462595c459b83023027645d57cccf; output-bytes=31; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [x] C3: cached bytecode and obsolete packaging metadata are gone, and imports use the preserved src packages
  CHECK: .venv\Scripts\python.exe -B results\repo_cleanup_20261025\check_installation.py
  EXPECT: CLEAN INSTALLATION OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=6af6a2d6dc1d77dd8c7cd8f8652fd502f71174b59883d8f0ea968cd6a6238dfa; exit=0; EXPECT=matched; output-sha256=764acdcbe50a061e81e6db018511ea648c6298ae7e6de89fabcfadf2f104aa2b; output-bytes=78; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [x] C4: package, replay and comparison smoke tests pass with bytecode and pytest-cache generation disabled
  CHECK: .venv\Scripts\python.exe -B -m pytest tests/test_glove_pipeline.py tests/test_joint_frames.py tests/test_replay_core.py tests/test_angle_coverage.py tests/test_glove_camera_pilot_check.py -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=d87c07a6738a49ee45d12a862d5638cab1f0bb872f70f57feddf7885196156c4; exit=0; EXPECT=matched; output-sha256=ef197e4f7764c9009827ec1e1cbc4dfaadc562de72bafc2fdedb936d5f9ce613; output-bytes=184; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries

- [x] C5: monitored original dataset files retain the forensic baseline hashes
  CHECK: .venv\Scripts\python.exe -B scripts\leap\replay_verify.py --check-dataset-unchanged "C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\baseline_sha256.json" "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized"
  EXPECT: DATASET UNCHANGED
  EVIDENCE: automatic-evidence=v1; definition-sha256=80f469e5a222db4e7cfb0bd953e55c0dc0660246ded16a50d06ae05e649cd7a8; exit=0; EXPECT=matched; output-sha256=d38204bbde96ec4916ca82f85f1b0410afb5c8aed343720ebd01558ce2ad3073; output-bytes=189; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=a581c9614ba6/38 entries
