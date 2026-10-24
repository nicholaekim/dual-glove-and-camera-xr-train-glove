# Gates: camera-clock erratum for the timing columns of the organized grasp dataset

OWNS: scripts/leap/camera_clock_erratum.py, tests/test_camera_clock_erratum.py, docs/replay_fidelity.md, results/replay_fidelity/camera_clock_erratum/**, GATES_camera_clock_erratum.md

Scope: a derived erratum, not a dataset change: for each of the 115 static takes and 52 transitions of `grasp set 2026-10-08 organized`, the quantities the metadata gives on the writer clock (`wall_time`) are recomputed on the camera clock (`timestamp_us`, falling back as `leap_hand.replay_core.build_timeline` does) and written beside the metadata values with their differences and the largest writer stall inside each interval, under results/replay_fidelity/camera_clock_erratum/ with a README and a summary; the dataset itself is not modified and nothing is committed.

- [x] E1: the erratum script's tests pass on a synthetic dataset directory (a writer stall inside the static interval shows as a camera-vs-writer difference over one frame period and is flagged; a take without a stall is not flagged; transition instants are matched to frames on the capture clock within the metadata's 1 ms rounding; blank cells for an invalid transition; output inside the dataset refused; a recording that cannot be loaded fails the run)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_camera_clock_erratum.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=4e6c5c0679cfac3a49f64a67322f96d3181608c4fd9b45e527a0deb895fa54a4; exit=0; EXPECT=matched; output-sha256=a3dab916007bde33faf1632f93cff8c1fce8532e08da4b4526d2f82d30d6b027; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] E2: the real-data test runs (not skipped) and reproduces p3_circular_precision_thumb_2_fingers take 4: static interval frames 151749-151855, 1.999 s on the writer clock, 1.178128 s on the camera clock, declared 2.000 s, largest writer stall about 0.807 s between frames 151852 and 151853
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_camera_clock_erratum.py -q -k real_data
  EXPECT: /[1-9][0-9]* passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=9941e5f1fc3de6a70c5daef77fa8deab77661106a13d8bc13bd9d700a396b573; exit=0; EXPECT=matched; output-sha256=32f85405dbce2ce3e2e0c716befec4a013b6af8894f93b5575371e9f2dd291e8; output-bytes=114; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] E3: the script runs over the whole dataset and writes static_takes.csv (115 rows), transitions.csv (52 rows), README.md and summary.md under results/replay_fidelity/camera_clock_erratum/
  CHECK: .venv\Scripts\python.exe scripts\leap\camera_clock_erratum.py --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized" --out results\replay_fidelity\camera_clock_erratum
  EXPECT: /ERRATUM OK: static=115 transitions=52/
  EVIDENCE: automatic-evidence=v1; definition-sha256=63f4da6f5c161b9463b4e30465eed77c3eb16f2bb896ce433dc36889a73288fc; exit=0; EXPECT=matched; output-sha256=937e3b7a07a1271527072682d55c1579e745a40d9ef72fb338fd6996b82f6574; output-bytes=126; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] E4: the dataset is unchanged: every file of `grasp set 2026-10-08 organized` listed in the forensic baseline hashes to its baseline SHA-256 after the erratum run
  CHECK: .venv\Scripts\python.exe scripts\leap\replay_verify.py --check-dataset-unchanged "C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\baseline_sha256.json" "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized"
  EXPECT: DATASET UNCHANGED
  EVIDENCE: automatic-evidence=v1; definition-sha256=7759d2ddcdacea4fe89e5943974205eccc4f2ac759fc1fcdd0fa5abf2afb0744; exit=0; EXPECT=matched; output-sha256=d38204bbde96ec4916ca82f85f1b0410afb5c8aed343720ebd01558ce2ad3073; output-bytes=189; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] E5: docs/replay_fidelity.md has a section pointing to the erratum folder and script
  CHECK: .venv\Scripts\python.exe -c "import sys; t = open('docs/replay_fidelity.md', encoding='utf-8').read(); ok = 'camera_clock_erratum' in t and 'scripts/leap/camera_clock_erratum.py' in t; print('DOCS POINT TO ERRATUM' if ok else 'DOCS MISSING POINTER'); sys.exit(0 if ok else 1)"
  EXPECT: DOCS POINT TO ERRATUM
  EVIDENCE: automatic-evidence=v1; definition-sha256=276f59e78e71e0641d90d0bdacef5e3c31c8a78c81e6e736a93bf562cf23f3b0; exit=0; EXPECT=matched; output-sha256=5cb8a4fc33d1c479e1e34d685878968ee5996c45e85cf61fd372556ae169eb56; output-bytes=23; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] E6: the whole test suite still passes
  CHECK: .venv\Scripts\python.exe -m pytest tests -q -x
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=82d9daa730e6f68dcc71f03b8a43c11ff501ce9ad583b6a269b29ac693626b8b; exit=0; EXPECT=matched; output-sha256=135383cb64ba1726d8f80704d4b6403fa3d5fc43cf12d466ac4f9bd6e17c8531; output-bytes=1666; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] E7: nothing is committed: HEAD is still the commit the task started from (05b0bd4)
  CHECK: git log --oneline -1
  EXPECT: /^05b0bd4 /
  EVIDENCE: automatic-evidence=v1; definition-sha256=77ccb0506b347f1471bb9d41a461ed8e7cb2ea94fd567d48ebf6f7b6f3e7778d; exit=0; EXPECT=matched; output-sha256=dd7d42455976dbfa7d526b72c54b4f9402361fb6e7f6af74121ad860eea1ea8c; output-bytes=253; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] E8: the erratum's README.md and summary.md were read in full: the README states the method, the three clocks and their roles, how each writer-clock instant was mapped to a frame, the rounding, and that the dataset itself is unchanged; the summary gives the distribution of the differences and lists the takes whose static interval differs by more than one frame period; both tables were opened and spot-checked against the probe numbers for thumb_2_fingers take 4 and p1_cylindrical transition take 4
  EVIDENCE: 2026-10-23, read in full by the session that wrote them, after the run of 21:25. README.md: what-this-is paragraph (derived tables, dataset opened read-only, output inside it refused, the --check-dataset-unchanged command, metadata left as it is); the three clocks with their roles; the build_timeline fallback order with clock_source/clock_fallback; hand selection; static-take columns (recording duration, the interval frames, declared vs writer span vs camera span, start/end origins, the stall columns); transition columns (the measured capture-clock premise, nearest frame by capture_time, the cue carried by the first frame captured after it from the unrounded cue_wall_time, duration/writer span/stall, blanks for invalid timings); rounding 1e-6 s / 1e-3 ms. summary.md: verdict line, the two static distribution tables, the 62 flagged takes listed with metadata/writer/camera/difference/stall, the 20 clock-only takes, the transition tables, the capture-vs-wall residual evidence (0.497 ms vs 20.608 ms). static_takes.csv row p3_circular_precision_thumb_2_fingers take 4: 107 frames 151749-151855, metadata 2.0, writer span 1.999331, camera 1.178128, difference -0.821872, stall 0.80744 s between 151852 and 151853 (camera 10.809 ms), recording 4.972 metadata / 4.977819 camera / metadata camera_clock_duration_s 4.978; matches the read-only probe of the same frames. transitions.csv row p1_cylindrical take 4: cue -0.001 metadata / -0.000753 camera (frame 803), movement start frame 871 (capture residual -0.139 ms, wall residual -0.143 ms) at 0.754881 s, settled frame 943 (-0.066 ms / +2.282 ms) at 1.554974 s, duration 0.8 metadata / 0.800093 camera / 0.813517 writer span, stall 0.037139 s between 882 and 883; matches the probe. Not checked by eye: the other 113 static rows and 51 transition rows beyond the distributions.
