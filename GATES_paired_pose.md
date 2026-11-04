# Gates: the paired-pose tool (one glove + camera take -> photo, Leap model, glove model each nearest the photo's moment, the same 21 points from both sensors in one frame, and where they disagree)

OWNS: src/leap_hand/paired_pose.py, scripts/paired_pose_compare.py, tests/test_paired_pose.py, src/leap_hand/protocol.py (still capture sidecar helpers only), scripts/leap/camera_view.py (still capture sidecar only), scripts/record_simultaneous.py (sidecar moved with the still, its fields copied into the meta), tests/test_camera_view.py (sidecar tests only), docs/paired_pose.md, GATES_paired_pose.md

Scope: `scripts/paired_pose_compare.py <session>` reads a `record_simultaneous.py --camera leap` session and, for every accepted take with both JSONL files, takes the PHOTO's moment (meta `still_capture_time`, else the still file's write time), picks each sensor's frame nearest that moment by capture_time within `--max-dt-ms`, writes both frames in the professor's text format through the one shared exporter, expresses the 21 MediaPipe-order points of both in millimetres with the wrist as origin and the axes of `cam_hand.fusion.palm_basis` built on each sensor's own hand (no fit, no scaling, no lag correction), and writes the per-point distances, the bone lengths, a per-take notes file, a three-panel figure (still, Leap model, glove model in the same palm frame), a session `comparison.csv` that lists refused takes with their reason, `figures.pdf` and `README.md`. The camera viewer writes `<still>.json` with the image frame's capture time; the recorder moves it with the still and copies it into the meta. Coordinates only, no joint angles.

Evidence below was recorded by running each CHECK by hand (Git Bash, repo root, 2026-11-03), not through gate-check.mjs: approving its commands is the user's step. Lint: `gate-lint.mjs GATES_paired_pose.md` printed LINT OK (4 warnings, both manual gates).

- [x] G1: the tool's functions and the script are pinned by tests on synthetic hands and synthetic sessions: frame_nearest picks the nearest frame on capture_time and refuses beyond max_dt, and falls back to wall_time with the clock named; the palm-frame coordinates are invariant to any rotation + translation of the input (1e-9); a mirrored hand flips same_handedness while a rotated copy does not; compare_points keeps WRIST out of the statistics; take_moment follows the priority meta capture stamp -> still file write time -> refused; the written camera.txt / glove.txt parse back through cam_hand.prof_format.parse_text with 21 landmarks; a take without a glove file or a still is listed in comparison.csv with its reason
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_paired_pose.py -q -k "not real_day2"
  EXPECT: /\d+ passed, 1 deselected/
  EVIDENCE: 2026-11-03, "12 passed, 1 deselected in 4.53s". Written first; the first run failed at collection (ImportError, the module did not exist yet). One test was wrong and was corrected: the no-moment take still carried a capture stamp, so it was a valid moment; it now has neither.

- [x] G2: the real-data test runs (not skipped) on recordings\sync_day2 take fist_left_take1_20260920_195141: the whole script into a temporary folder, every output present, comparison.csv one row with moment_source "still file write time (no capture stamp in this take)"
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_paired_pose.py -q -k real_day2
  EXPECT: /\b1 passed, \d+ deselected/
  EVIDENCE: 2026-11-03, "1 passed, 12 deselected in 2.29s" (passed, not skipped).

- [x] G3: the still capture sidecar: built from the image frame (capture_time from the LeapC frame age, else saved_at with capture_time_source "saved_at"), moved with the still by move_still (accepted rename and rejected path), copied into the meta as still_capture_time / still_timestamp_us / still_frame_id / still_capture_time_source and null without a sidecar; the existing camera-view and recorder tests stay green; camera_view.py compiles
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_camera_view.py tests/test_record_simultaneous_hints.py -q
  EXPECT: / passed/
  EVIDENCE: 2026-11-03, "34 passed in 16.98s" (30 existing + 4 new). Limit: the viewer's own new lines (reading the image event's LeapC timestamp and frame id, writing the sidecar after the JPEG) need the camera and are covered by compilation (test_viewer_script_exists_and_compiles) only, not run.

- [x] G4: the four rehearsal takes of day 2 (fist, open palm, pinch, peace; left) run into the default results folder
  CHECK: .venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\sync_day2 --takes fist_left_take1_20260920_195141 open_palm_left_take1_20260920_195104 pinch_left_take1_20260920_195513 peace_left_take1_20260920_195357
  EXPECT: /PAIRED POSE DONE: takes=4 processed=4 refused=0/
  EVIDENCE: 2026-11-03, "PAIRED POSE DONE: takes=4 processed=4 refused=0". Medians 23.0 / 17.9 / 15.0 / 16.2 mm, largest 47.4 (THUMB_TIP) / 26.6 (MIDDLE_FINGER_TIP) / 82.2 (INDEX_FINGER_TIP) / 72.7 mm (RING_FINGER_TIP), every |dt| <= 11.1 ms, same handedness in all four.

- [x] G5: the whole day-2 session runs into the default results folder; every accepted take is either processed or listed with its reason
  CHECK: .venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\sync_day2
  EXPECT: /PAIRED POSE DONE: takes=60 processed=\d+ refused=\d+/
  EVIDENCE: 2026-11-03, "PAIRED POSE DONE: takes=60 processed=60 refused=0", run after G4 so the default folder holds all 60; "13 rejected attempt(s) under rejected/ not processed (--include-rejected)".

- [x] G6: the outputs on disk are complete and consistent, re-read from the files: per processed take camera.txt and glove.txt with one 21-landmark block each, points_common_frame.csv with 21 rows, bone_lengths.csv with 20 rows, notes.txt, figure.png; comparison.csv one row per take; figures.pdf one page per processed take; README.md naming units, origin, axes, moment rule, Palm point, template bones, lag, labels
  CHECK: .venv\Scripts\python.exe scripts\paired_pose_compare.py --check-outputs results\paired_pose\sync_day2
  EXPECT: /OUTPUTS OK: takes=60 /
  EVIDENCE: 2026-11-03, "OUTPUTS OK: takes=60 processed=60 pages=60". Negative control on a scratch copy of one take with glove.txt deleted and "error" appended to its README: "OUTPUTS NOT OK: ... no glove.txt / README.md ... uses error/accuracy wording", exit 1; the same folder before the damage printed OUTPUTS OK.

- [x] G7: one figure.png looked at by eye at full size (Read of the PNG): the uncut still, both models in the same frame with identical limits, letters, wrist origin, 20 mm bar, labels and suptitle readable; anything unreadable fixed and re-rendered
  EVIDENCE: 2026-11-03. First render (pinch_left_take1): the label box covered the Leap "M" and the photo title sat lower than the model titles. Second (peace_right_take1): the box covered the wrist marker on a right hand. Fixed by moving the sensor label out of the plot to a line under each model title and placing the photo title on the same line as the model titles. Final renders read in full: peace_right_take1 and fist_left_take1 (1800 x 600): still uncut, both model panels with the same limits and scale, palm view and side view, T I M R L past the tips (inside the palm outline in a fist, where the tips curl back), wrist circle with "wrist = origin", 20 mm bar, axis labels naming the direction, the labels line, a three-line suptitle with pose, take, hand, moment and both dt values.

- [x] G8: docs/paired_pose.md states what the tool is for, the three commands (no hardware on day 2, the new session, where the outputs go) and the conventions list; the README template in the script states every convention of the brief in its own words
  EVIDENCE: 2026-11-03. docs/paired_pose.md: purpose; three commands (day-2 rehearsal, the new session, the output check), each naming its hardware setup; where outputs go; nine convention bullets. The generated README.md read in full: what each file is; mm; wrist origin; the palm_basis recipe with landmark numbers and names; the moment rule with both source labels; same-moment rule and the 50 ms limit; Palm point 0 not compared; WRIST excluded from statistics; glove template bones, tracker bones vary, no scaling; glove lag per hand not corrected; labels; handedness rule; wording; coordinates only; an automatic warning when stills are the whole 768 x 768 frame (60 of 60 on day 2). Both bone claims measured on this run's own bone_lengths.csv, not taken from the brief: the glove's 20 bones vary by 0.000 mm across the 60 takes, the tracker's by 5.8 to 23.5 mm per bone (left index MCP-PIP 22.7 to 33.0 mm, glove 36.9).

- [x] G9: nothing is committed and no file outside OWNS was changed by this work: HEAD is still b9023ef
  CHECK: git log -1 --format=%h
  EXPECT: /^b9023ef/
  EVIDENCE: 2026-11-03, "b9023ef". git status: 56 paths = the 47 the other sessions had uncommitted at the start + this work's 4 modified (camera_view.py, record_simultaneous.py, protocol.py, test_camera_view.py, none of which were modified before) and 5 new.

Corrections from the independent review (2026-11-03, second pass): G11 to G14. Run by hand as above; the full suite (G10) was not re-run for them, only tests/test_paired_pose.py.

- [x] G11: the palm frame refuses a degenerate hand and checks its basis: an angle between x = M - W and P - I under 10 degrees (0.0 and 5.0 tested) or a missing/non-finite W, M, P or I raises DegeneratePalm with "degenerate palm frame (<sensor>, angle N deg)" (or the missing landmark's name), which the script writes as the take's status; on rotated copies of a hand every entry of |B.T B - I| is under 1e-9, and check_orthonormal (called inside to_palm_frame) raises on a basis 1e-6 off
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_paired_pose.py -q -k degenerate_palm
  EXPECT: /\b1 passed, \d+ deselected/
  EVIDENCE: 2026-11-03, "1 passed". Confirmed by reading cam_hand/fusion.py: palm_frame computes x = unit(M - W), n = unit(x × (P - I)), y = unit(n × x); palm_basis stacks the columns x, y, n; to_palm_frame multiplies the wrist-centred points by it, so local coordinates are dot products with x, y, z. No day-2 take was refused by the guard (60 of 60 processed).

- [x] G12: handedness is a diagnostic with three outcomes: a rotated + translated copy gives "same", a mirror (y flipped) gives "opposite (suspected mirrored tracker fit)", a flat hand (every point in the palm plane, margin 0) gives "indeterminate (margin under 5 mm)" on either side; comparison.csv carries handedness, glove_z_margin_mm, camera_z_margin_mm; the day-2 minimum margin re-measured on this run
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_paired_pose.py -q -k handedness_is_same
  EXPECT: /\b1 passed, \d+ deselected/
  EVIDENCE: 2026-11-03, "1 passed". Day 2 (G14 run): handedness "same" 60 of 60; smallest |margin| camera 5.3 mm (open_palm_left_take5_20260920_195132, glove there -8.1 mm), glove 7.9 mm (open_palm_right_take3_20260920_195720); both smallest on open palms, as the README says.

- [x] G13: the stillness gate: window_speed_mm_s is 0 for a still sequence and for the whole hand carried at 520 mm/s (translation removed), 40 x 20/21 mm/s when the 20 non-wrist points move at 40 mm/s, NaN with one frame; stillness() gives "still", "moving (glove)", "moving (camera)", "moving (both)" around STILL_MM_S = 10 (transition.POSE_SETTLE_SPEED_MM_S, reused); the day-2 distribution recorded
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_paired_pose.py -q -k stillness_on
  EXPECT: /\b1 passed, \d+ deselected/
  EVIDENCE: 2026-11-03, "1 passed". Day 2: 8 of 60 "moving": "moving (camera)" 6 (open_palm_right_take1 16.6, peace_right_take1 18.6, take2 72.2, take3 55.5, take4 24.4, pinch_left_take2 10.2 mm/s), "moving (glove)" 1 (fist_right_take4, glove 91.0 mm/s), "moving (both)" 1 (index_point_right_take2, glove 27.0, camera 48.0 mm/s); 52 "still". Medians glove 0.1, camera 2.3 mm/s.

- [x] G14: the whole day-2 session reruns with the new columns and the output check passes; the check now also requires the new README phrases (label, not physically registered, WRIST and PALM, the axis formula, degenerate palm frame, ARRIVAL, the landmark section, POSE_SETTLE_SPEED_MM_S, no scaled table, the three handedness values), the notes phrases (label, timing uncertainty, stillness, transformation, handedness, statistic basis) and valid handedness / stillness values
  CHECK: .venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\sync_day2 && .venv\Scripts\python.exe scripts\paired_pose_compare.py --check-outputs results\paired_pose\sync_day2
  EXPECT: /OUTPUTS OK: takes=60 processed=60 pages=60/
  EVIDENCE: 2026-11-03, "PAIRED POSE DONE: takes=60 processed=60 refused=0", then "OUTPUTS OK: takes=60 processed=60 pages=60". Largest |dt| 32.3 ms; moment_source "still file write time (no capture stamp in this take)" on all 60. tests/test_paired_pose.py: "16 passed" (13 existing, one of them updated from the removed same_handedness column to handedness + stillness + check_outputs, and 3 new). figure.png of index_point_right_take2 read at full size: the new titles and the label in the suptitle are readable.

Round 2 of the review (wording only, no new logic): G15.

- [x] G15: the timing label "arrival-time matched, held-pose comparison" stands beside "wrist-centred, independently palm-aligned shape disagreement" in README.md, every notes.txt and the figure suptitle, with the sentence "Physical simultaneity of the two selected frames is not verified; ..." in README.md and every notes.txt; no output says "same moment" or "same instant" (the professor's "all showing the same moment" survives only in the module/docs as a quote of her request); docs/paired_pose.md step 2 is the enforced first-take gate (2.1 to 2.8, gloves on + camera, four column checks plus figure.png by eye, repeat the take on any failure); --check-outputs now also requires the label and the sentence (README_MUST, NOTES_MUST)
  CHECK: .venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\sync_day2 && grep -l "arrival-time matched, held-pose comparison" results/paired_pose/sync_day2/README.md results/paired_pose/sync_day2/*/notes.txt | wc -l
  EXPECT: /^61$/
  EVIDENCE: 2026-11-04, rerun "PAIRED POSE DONE: takes=60 processed=60 refused=0", grep 61 files (README + 60 notes); the simultaneity sentence also in 61 files; grep -il "same moment\|same instant" over README and notes: 0 files; "OUTPUTS OK: takes=60 processed=60 pages=60"; tests/test_paired_pose.py "16 passed". The suptitle is drawn in the PNG and not greppable: read by eye on peace_right_take2_20260920_200035/figure.png, its third line reads "wrist-centred, independently palm-aligned shape disagreement; arrival-time matched, held-pose comparison". The first render overlapped the panel titles (four suptitle lines); the titles and plots were moved down (top 0.78 -> 0.75, titles 0.865 -> 0.835) and the re-render is clean.

- [x] G10: the whole test suite passes (about 23 minutes), re-run after the G11 to G15 edits
  CHECK: .venv\Scripts\python.exe -m pytest tests -q -x
  EXPECT: / passed/
  EVIDENCE: 2026-11-04, run by the planning session after all review edits: "1559 passed, 5 skipped in 1370.44s (0:22:50)", exit 0 (1556 after the build + 3 new). First run after the build, 2026-11-03: "1556 passed, 5 skipped in 1373.56s (0:22:53)", exit 0 (1539 before this work + 17 new). Independent review (ChatGPT project "xr training (work study)", thread "Step review comparison tool"): round 1 seven corrections (G11 to G14), round 2 three corrections (G15, the first-take gate, this re-run), round 3 VERIFIED on 2026-11-04.
