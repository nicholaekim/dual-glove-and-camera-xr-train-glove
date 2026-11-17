# Gates: what MuJoCo and MyoSuite can do for us (a colleague's links, 6 Nov 2026)

OWNS: scripts/leap/myosuite_hand_limits.py, tests/test_myosuite_constraints.py, docs/myosuite_constraints.md, results/myosuite_constraints/**, .gitignore, README.md, scripts/leap/angle_coverage_report.py, scripts/leap/camera_clock_erratum_report.py, GATES_myosuite.md

Scope: find out, on this Windows machine, what the two simulation tools a colleague sent (MuJoCo and MyoSuite) can do for the glove-and-camera project: install them in a repo-local venv, load MyoSuite's musculoskeletal hand with MuJoCo, dump every hand joint's biomechanical range in degrees, set those ranges beside Paper 3's Table 5 ranges and the glove's measured spans under the repo's 24 angle names, and write a short note saying what each tool is, what we can do with it (compare captured poses with a simulated hand, compare constraints, an anatomical filter), what we cannot, and the next step. No hardware is run; nothing is committed.

- [x] M1: MuJoCo and MyoSuite install in `.venv-myo` on this machine, the MyoSuite hand model compiles, and 100 simulation steps run with finite joint positions
  CHECK: .venv-myo\Scripts\python.exe scripts\leap\myosuite_hand_limits.py --check
  EXPECT: /MYOHAND OK: mujoco=\S+ myosuite=\S+ joints=\d+ steps=100 finite=yes/
  EVIDENCE: automatic-evidence=v1; definition-sha256=5b8449a5f4548af1c2cf1df395f00e63cde463bca505efcdf6954f7757516dbc; exit=0; EXPECT=matched; output-sha256=18ec6e5c8565fd0a744966f47140d6f95acb6409d714ab8a19258b7eede25b3d; output-bytes=437; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] M2: `results/myosuite_constraints/myohand_joint_limits.csv` has one row per hand joint (name, limited, min_deg, max_deg, span_deg, axis, parent body), and `results/myosuite_constraints/comparison_24_angles.csv` has exactly the repo's 24 angle names (joint_frames.DOF24_NAMES, same order) with the MyoSuite joint matched to each, its range in degrees, Paper 3's Table 5 range and hyperextension as printed, and the glove's measured span from the pilot audit, with an empty cell wherever a source has nothing for that angle
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_myosuite_constraints.py -q -k "csv"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=56ed27657fca66de8f880cbab8b17ceadbd3bbb31ba43bece5a30f16c15092e9; exit=0; EXPECT=matched; output-sha256=0fd32bad2533d9b757d95c220f8536b852a942e9f894dd41c6067f5131016053; output-bytes=114; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] M3: `docs/myosuite_constraints.md` says, in its headings, what the two tools are, what we can do with them, what we cannot, what the joint comparison shows, how to reproduce it (one command) and the next step; and every MyoSuite number it quotes is the one in the CSV
  CHECK: .venv\Scripts\python.exe -m pytest tests\test_myosuite_constraints.py -q -k "document"
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=6200f3d72cd1671bcaa5a8ecc3f8dd3aa1d3d2592770fb2f4f1aee7362330743; exit=0; EXPECT=matched; output-sha256=490d4addf4bece6be8d3cab91e42ea36483a4c5edc538466850fd50515d49add; output-bytes=114; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] M4: the whole test suite passes with the new test
  CHECK: .venv\Scripts\python.exe -m pytest tests -q -x
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=82d9daa730e6f68dcc71f03b8a43c11ff501ce9ad583b6a269b29ac693626b8b; exit=0; EXPECT=matched; output-sha256=015aa3ca7b2d4029c66daf6dce491ec910214ff110a6b8ac19921e41282e1b8a; output-bytes=1909; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] M5: nothing is committed: HEAD is still 95b9040
  CHECK: git log -1 --format=%h
  EXPECT: /^95b9040/
  EVIDENCE: automatic-evidence=v1; definition-sha256=0e45c053f9dc1820407a11848e098f42669cbfc5dd88e3f01db015f4bc5d60f2; exit=0; EXPECT=matched; output-sha256=314e0e43b315f489baacf140ed4a172182d39ac5702bc9fb6e1dd9b68bdac6f0; output-bytes=8; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries
