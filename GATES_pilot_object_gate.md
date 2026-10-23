# Gates: the object gate (typed object ID plus an unobstructed audit still before every object grasp)

OWNS: scripts/pilot_run.py, tests/test_pilot_run.py, scripts/pilot_compare.py, tests/test_pilot_compare.py, docs/pilot_hand_on_object.md

Scope: twice on 2026-10-23 the operator mimed the object runs instead of holding the objects, and nothing in the record could show it. From now on, for B1, C and B2, `pilot_run.py` does not start a grasp's takes until (1) the operator, with the named object held in the LEFT hand over the camera, has typed that object's ID (OBJ-CYL, OBJ-BALL or OBJ-PEN; three tries, then the run stops), and (2) one raw infrared audit still of the hand with the object has been taken by `scripts/leap/ir_snapshot.py` into `<run>\audit\` (both eyes, no skeleton overlay, with its sidecar naming the hands the tracker saw at that instant); a missing still stops the run. Confirmations are appended to `<run>\audit\confirmations.txt`. `pilot_compare.py` counts the audit stills and the confirmations per run x grasp and prints them in the report. Mimed runs have no gate. `--print` runs nothing and asks nothing.

- [x] O1: the gate is pinned by tests (wrong ID three times -> stop, no recorder call, no confirmation line; q -> stop; right ID -> confirmation line appended with time, grasp and ID, the snapshot command issued into <run>\audit with the grasp and ID in the label, then the recorder; a snapshot that leaves no new *_L.png -> stop before the recorder; A and A2 never prompt; --print prompts nothing and runs nothing; the USE line before the prompt names the object line verbatim)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_pilot_run.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=2873a2d500f999771141b4427e7c9acc970d7772314e4d5937e6c886d2a7587d; exit=0; EXPECT=matched; output-sha256=6bd0062435f5728e8f16581f191611551b769a9ac8cae6ea7857b59d6907c84f; output-bytes=101; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] O2: pilot_compare.py reports per run x grasp the number of audit stills and of typed confirmations (tests on synthetic audit folders; absent folder -> 0)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_pilot_compare.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=74b20741281b2c2e28179eeedd5b7e5032715417408b63ff8a7313069be63e51; exit=0; EXPECT=matched; output-sha256=2e9c52b35f48c4a510bdf132d1712a4ad671c09b7595baa547253999dd50f110; output-bytes=102; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] O3: the whole test suite still passes
  CHECK: .venv\Scripts\python.exe -m pytest tests -q -x
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=82d9daa730e6f68dcc71f03b8a43c11ff501ce9ad583b6a269b29ac693626b8b; exit=0; EXPECT=matched; output-sha256=e57cf2cf325edafbca6801374f7cb2d558ca9a4b9e8304b6b142e9bda1335ac1; output-bytes=1423; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] O4: on the first attempt's folder, `B1 --print` shows, for each grasp, the audit snapshot command with the grasp and object ID in its label before the recorder line, and says the typed confirmation is required
  CHECK: .venv\Scripts\python.exe -c "import subprocess, sys; r = subprocess.run([sys.executable, r'scripts\pilot_run.py', 'B1', '--print', '--root', r'recordings\pilot_hoo_2026-10-23_attempt1'], capture_output=True, text=True); print(r.stdout); sys.exit(r.returncode)"
  EXPECT: /ir_snapshot\.py[^\n]*cylindrical_object_OBJ-CYL[\s\S]*record_simultaneous\.py[^\n]*--poses cylindrical_object/
  EVIDENCE: automatic-evidence=v1; definition-sha256=9c893513b88c5609e8d7f6bb24ad432f70659cdc2a1c4826f3d2a32e1b868bab; exit=0; EXPECT=matched; output-sha256=9aac75d807baaa4cd3b1b3dff3fc243dd671a446ba33d191491b1ee102ea1a61; output-bytes=2024; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] O5: docs/pilot_hand_on_object.md states the gate (what the operator types, the audit still, where both are kept, that a grasp cannot start without them), that "Allow images" must be on in the Ultraleap Control Panel (step 3), and that step 3 uses the both-gloves launcher when XR Trainer streams both; the gate was verified in the ChatGPT thread
  EVIDENCE: 2026-10-23. docs/pilot_hand_on_object.md: section "Amendment of 2026-10-23, second: the object gate" (why; the typed confirmation with three tries and q, kept in <run>\audit\confirmations.txt; the raw infrared audit still by ir_snapshot.py into <run>\audit\ with its sidecar; no picture stops the run; the comparison's audit counts; mimed runs ungated); step 2c (camera-only rehearsal of the still in the throwaway folder recordings\pilot_hoo_gate_test, never the pilot root, as the reviewer required); step 3 ("Allow images" ON; pilot_step3_gloves_both.cmd because XR Trainer streams both gloves, the left-only launcher only with the right glove physically disconnected); step 5 names the gate for steps 5 to 7. ChatGPT thread 'Open source hand object models', 2026-10-23: the gate reported with its tests and file names; one point (rehearse in a throwaway root) applied; writing the confirmation before the still confirmed as correct; verdict VERIFIED.
