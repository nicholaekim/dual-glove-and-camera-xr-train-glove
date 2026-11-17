# Gates: the professor's live demo of the paired-pose pipeline (one pose, live, about five minutes, with a no-hardware fallback)

OWNS (all new files): GATES_prof_demo.md, launchers/prof_demo.cmd, launchers/prof_demo_mock.cmd, C:\Users\nkim2\OneDrive\Desktop\prof_demo.cmd, C:\Users\nkim2\OneDrive\Desktop\prof_demo_mock.cmd, C:\Users\nkim2\OneDrive\Desktop\xr trainer\prof demo 2026-11 runbook.md and its .pdf. Output folders written by running them: recordings\prof_demo, recordings\prof_demo_mock, results\paired_pose\prof_demo.

Not touched (another session edits them): GATES_paired_pose.md, docs/paired_pose.md, docs/sessions/, results/paired_pose/ten_poses_left (read only: selection.csv and the accepted figures), the hand-in folder `paired poses 2026-11-05`, scripts/paired_pose_compare.py. Nothing is committed.

Scope: `prof_demo.cmd [pose]` (gloves on + camera, left hand; pose one of pinch (default), open_palm, fist, peace) prints the setup checklist, runs `scripts/leap/check_setup.py --seconds 3`, pauses, records one take into `recordings\prof_demo` with `--first-take auto`, runs `scripts/paired_pose_compare.py recordings\prof_demo` and opens that new take's `figure.png` with `start ""`. If the recorder or the tool produced no figure for the new take, it says so and opens nothing older. `prof_demo_mock.cmd [pose]` (no hardware) runs the same flow with `check_setup.py --mock`, `--mock-glove --mock-leap --no-pose-check` into `recordings\prof_demo_mock` (beside the real folder, never in it), says on screen that a mock take has no photo so the flow ends at the recorder, and opens the pre-recorded figure of the pose's accepted take, read from `results\paired_pose\ten_poses_left\selection.csv` column `accepted_take`.

Deviation from the brief, decided while building: the mock launcher records with `--prep 0 --duration 1`, not 5 s. The mock camera swaps hands every 4 s (`MockLeapStream cycle_seconds=4.0`), so with a 5 s hold all four attempts were refused ("the tracker let go and re-acquired the hand"); these are the flags of the ten-pose mock rehearsal (GATES_paired_pose G18). The real launcher keeps `--duration 5`.

Finding: `record_simultaneous.py` exits 0 even when the take failed (first mock run: "0/1 takes complete", and the launcher's `if errorlevel 1` line did not fire). The real launcher therefore does not trust the exit code alone: it compares the newest take stem before and after recording (G8).

CHECK commands run in Git Bash from the repo root, by hand, 2026-11-06.

- [x] G1: all four launchers are CRLF batch files
  CHECK: file launchers/prof_demo.cmd launchers/prof_demo_mock.cmd /c/Users/nkim2/OneDrive/Desktop/prof_demo.cmd /c/Users/nkim2/OneDrive/Desktop/prof_demo_mock.cmd
  EXPECT: four lines, each "with CRLF line terminators"
  EVIDENCE: 2026-11-06, four lines "DOS batch file, ASCII text, with CRLF line terminators".

- [x] G2: the Desktop copies are byte-identical to the repo launchers
  CHECK: cmd //c "fc /b launchers\prof_demo.cmd C:\Users\nkim2\OneDrive\Desktop\prof_demo.cmd" ; cmd //c "fc /b launchers\prof_demo_mock.cmd C:\Users\nkim2\OneDrive\Desktop\prof_demo_mock.cmd"
  EXPECT: /FC: no differences encountered/ twice
  EVIDENCE: 2026-11-06, "FC: no differences encountered" for both pairs (after the last edit of the mock launcher).

- [x] G3: the mock launcher, run with no hardware, reaches the fallback and opens the accepted pinch figure (pinch_left_take4_20261106_104549 in today's selection.csv; the launcher reads it from the file)
  CHECK: cmd //c "launchers\prof_demo_mock.cmd pinch" < /dev/null
  EXPECT: /FALLBACK FIGURE: .*ten_poses_left\\pinch_left_take4_20261106_104549\\figure\.png/ and the "no photo" message before it
  EVIDENCE: 2026-11-06, "Session summary (left hand): 1/1 takes complete", then "A MOCK TAKE HAS NO PHOTO, so the paired-pose tool refuses it: the live flow ends here, at the recorder.", then "FALLBACK FIGURE: C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\results\paired_pose\ten_poses_left\pinch_left_take4_20261106_104549\figure.png"; the image viewer opened. The figure was read at full size: photo of a thumb-index ring, Leap model with thumb at index, glove model with the thumb out, median 10.8 mm, largest 48.4 mm (THUMB_TIP). The first run (5 s hold) failed at the recorder as described above and still reached the same fallback line.

- [x] G4: the same for a second pose run from the Desktop copy, so the copy works from outside the repo and the lookup is per pose
  CHECK: cmd //c "C:\Users\nkim2\OneDrive\Desktop\prof_demo_mock.cmd fist" < /dev/null
  EXPECT: /FALLBACK FIGURE: .*ten_poses_left\\fist_left_take1_20261105_234414\\figure\.png/
  EVIDENCE: 2026-11-06, "PASS  mock pipeline ...", "fist 1/1 complete (1 attempt(s))", "FALLBACK FIGURE: ...\ten_poses_left\fist_left_take1_20261105_234414\figure.png".

- [x] G5: the fallback is not hard-coded: the mock launcher reads selection.csv / accepted_take and names no take stem
  CHECK: grep -c "selection.csv" launchers/prof_demo_mock.cmd ; grep -c "accepted_take" launchers/prof_demo_mock.cmd ; grep -cE "_take[0-9]" launchers/prof_demo_mock.cmd
  EXPECT: /^[1-9]/, /^[1-9]/, then 0
  EVIDENCE: 2026-11-06, "2", "1", "0".

- [x] G6: a pose outside the four is refused before any check or recording, on both launchers
  CHECK: cmd //c "launchers\prof_demo.cmd thumbs_up" < /dev/null ; cmd //c "launchers\prof_demo_mock.cmd thumbs_up" < /dev/null
  EXPECT: /not a demo pose/ twice, and no check_setup output
  EVIDENCE: 2026-11-06, twice '"thumbs_up" is not a demo pose. Use one of: pinch, open_palm, fist, peace. Nothing was recorded.' and nothing else before the pause.

- [x] G7: the real launcher carries the exact commands of the brief (read, not run: it needs the gloves and the camera)
  CHECK: grep -E "check_setup.py --seconds 3|--camera leap --hand left .*--takes 1 --first-take auto .*--out-dir %REC%|paired_pose_compare.py %REC%|start \"\"" launchers/prof_demo.cmd ; grep -n "^set REC" launchers/prof_demo.cmd
  EXPECT: four matching lines (check, recorder, tool, start), and REC = recordings\prof_demo
  EVIDENCE: 2026-11-06, the four lines "%PY% scripts\leap\check_setup.py --seconds 3", "%PY% scripts\record_simultaneous.py --camera leap --hand left --poses %POSE% --takes 1 --first-take auto --duration 5 --out-dir %REC% --note "professor live demo"", "%PY% scripts\paired_pose_compare.py %REC%", "start "" "%FIG%""; "10:set REC=recordings\prof_demo".

- [x] G8: the real launcher's "newest take" logic picks the take this run recorded and never an older one: with an unchanged folder it reports "no new take"
  EVIDENCE: 2026-11-06, the launcher's own `:newest` lines (copied by sed from launchers/prof_demo.cmd into a scratchpad harness) run through cmd.exe: recordings\prof_demo_mock pinch -> pinch_left_take2_20261106_105522 (newest of take 1 rejected and take 2); recordings\ten_poses_left pinch -> pinch_left_take4_20261106_104549; open_palm -> open_palm_left_take2_20261105_234642; a missing folder -> empty. The launcher sets BEFORE before recording and AFTER after; equal (also both empty) goes to "no new take", so an older figure is never opened. A take the tool refuses has no figure.png and goes to "the tool made no figure", which prints its comparison.csv row.

- [x] G9: the runbook exists as MD and PDF, the PDF is one page and every page was looked at (Read of the PDF)
  CHECK: .venv/Scripts/python.exe -c "import sys;from pypdf import PdfReader;print('pages', len(PdfReader(sys.argv[1]).pages))" "/c/Users/nkim2/OneDrive/Desktop/xr trainer/prof demo 2026-11 runbook.pdf"
  EXPECT: /pages 1/
  EVIDENCE: 2026-11-06, "pages 1" at 9.5 pt (10 pt gave 2 pages, so 9.5 was kept). Rendered by a scratchpad reportlab script from the MD; the page was read as an image: headings, numbered setup, timed sequence, fallback, three questions and the files line all present and legible, nothing cut.

- [x] G10: honest wording: every runbook line that says error or accuracy says it in the negative
  CHECK: grep -iE "error|accura" "/c/Users/nkim2/OneDrive/Desktop/xr trainer/prof demo 2026-11 runbook.md" | grep -viE "\bnot\b|never|don't|no " | wc -l
  EXPECT: /^0$/
  EVIDENCE: 2026-11-06, "0" (the only such line is "Do not say: error, accuracy, ground truth").

- [x] G11: every claim in the runbook matches docs/paired_pose.md and docs/sessions/ten_poses_left.md: left glove about 0.47 s, not corrected; frames nearest the photo on each sensor's own clock, simultaneity not verified; bent_wrist control sentence; what the glove cannot show (pinch closing, true spread, crossed fingers); camera tilted toward the operator in the ten-pose session; handedness "opposite (suspected mirrored tracker fit)" and the tracker letting go of the hand; neither sensor is the reference
  EVIDENCE: 2026-11-06, read line by line against both docs (read only). 0.47 s left, packet arrival, not corrected (paired_pose.md Conventions; session "How to read the numbers"). Nearest the photo's time on each sensor's own clock within 50 ms, physical simultaneity not verified, stillness reduces but does not remove (Conventions bullet 1). bent_wrist control (paired_pose.md "The ten poses"). Glove flexion only, template spread, no wrist, cannot show a pinch closing, a true spread or crossed fingers; a known inability is the disagreement under study (runbook step 11). Camera tilted toward the operator (session "Camera orientation"). Handedness string from src/leap_hand/paired_pose.py OPPOSITE; comparison.csv: 26 of 26 takes "same". 5 refused attempts: one lost hand, four let go and re-acquired (session "Selection and exclusions"); the runbook says "lose the hand, or let go of it and re-acquire it". Still cropped, nothing drawn, mirrored as the window shows it (Conventions). 48.4 mm at THUMB_TIP from comparison.csv for the accepted pinch take. Neither sensor the reference (Conventions last bullet).

- [x] G12: nothing committed: HEAD is still 95b9040
  CHECK: git log -1 --format=%h
  EXPECT: /^95b9040/
  EVIDENCE: 2026-11-06, "95b9040"; git status lists GATES_prof_demo.md, launchers/prof_demo.cmd and launchers/prof_demo_mock.cmd as untracked.

- [ ] G13 (the user's, gloves on + camera): `prof_demo.cmd` on the Desktop records one pinch and opens its figure.png
  EVIDENCE:
