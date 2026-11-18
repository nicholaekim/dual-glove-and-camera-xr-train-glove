# Gates: validation and measurement workflow for the six unavailable Paper 3 coordinates (thumb TMC pair, four finger CMC)

OWNS: src/leap_hand/reference_validation.py, scripts/leap/reference_validation.py, scripts/leap/record_validation.py, src/leap_hand/angle_coverage.py, src/leap_hand/replay_viewer.py, scripts/leap/replay_view.py, scripts/leap/replay_verify.py, tests/test_reference_validation.py, tests/test_record_validation.py, tests/test_angle_coverage.py, tests/test_replay_viewer.py, tests/test_replay_verify.py, docs/measurement_specification.md, docs/reference_validation.md, docs/validation_capture_protocol.md, docs/validation_handoff_codex.md, docs/angle_coverage.md, docs/tmc_estimate.md, README.md, GATES_reference_validation.md

Scope: a measurement specification for T_TMC_fe, T_TMC_aa, I_CMC, M_CMC, R_CMC, L_CMC that keeps Paper 3's model assumptions apart from what this dataset supports; a documented reference-measurement file format with an importer that rejects other hands, missing units, ambiguous conventions and incompatible calibrations; time alignment with stated synchronization uncertainty, frame/axis alignment, valid-sample matching that never crosses a tracking discontinuity, a mapping fitted on calibration trials and evaluated on held-out trials only; per joint and movement bias, MAE, RMSE, percentiles, repeatability, cross-talk, counts, exclusions and tested range; measurement status visible in the viewer, the exports and the coverage table; a live capture procedure with exact commands; the existing dataset's reconstruction unchanged. Physical accuracy, live capture and the finger CMC need equipment and a participant: they are abandoned here with the reason, never passed on synthetic data.

- [x] V1: the measurement specification names, for each of the six coordinates, the segments and axes, the reference posture, the rotation order, the signs and units, what the tracker supplies, the additional measurement required and how accuracy is checked, and keeps Paper 3's assumptions apart from what the dataset supports
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k spec -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=54d668eb5781e410559f900149fcba617cee9cbd17bdd91b2cb3d62f541613c8; exit=0; EXPECT=matched; output-sha256=c0ef450516f21c84744607f6183594ad2ab8aaeecec57c9b05ad7719f7bc81c8; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V2: a reference capture in the documented format imports, and every malformed one is refused with an actionable message (missing units, unknown units, ambiguous sign or order, unknown joint or movement, other hand, a derived-from-tracker file posing as a reference, missing columns, edited samples file)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k format -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=e916e1444fef131f98f0f88db335aa82f670e7a85c21549efb5d418872919cd8; exit=0; EXPECT=matched; output-sha256=d223f9948cc51f36ebbee7c2b2d49938fa6cbd62f2ca4bb927efe1c09e5f2add; output-bytes=117; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V3: synchronization recovers a known clock offset and drift from paired events (detected thumb-index taps included), states its uncertainty, refuses events that disagree beyond the declared uncertainty, and reports how the errors move when the clock is shifted by that uncertainty
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k sync -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=6578f1023c98e91c5f022a115f59a3d159c5adc839209b7930a348cecd53ceba; exit=0; EXPECT=matched; output-sha256=e2b0d8e6efad0a27bbe4c102eb9c96d6477e6f0bb37c6ea72db93ff4876380c0; output-bytes=116; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V4: hand, participant, session and calibration compatibility are enforced on import, on comparison and when a mapping is applied, each refusal naming what differs
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k compat -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=edbaf2360c7f6cb582772324e36319d80d6a442cca1e19844c39bc9b61418530; exit=0; EXPECT=matched; output-sha256=659722b4348ba06205b7140c46342ee79faa4e130979d2d7c546cbd870061347; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V5: frame and axis alignment recovers a known palm-to-anatomical rotation and a known metacarpal long-axis offset in both rotation orders and both hands, and refuses a fit the calibration samples cannot determine (one posture, one movement plane)
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k align -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=85cc30de4f0009818bf555a1f70cbe0599317e5e193a6986796218e1bebccfed; exit=0; EXPECT=matched; output-sha256=a652ec91e61f697cc3c3249a75297304e3a056f932dd2441e5dd1c94cf66059b; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V6: the comparison reports an injected bias, gain and cross-talk at their known values (signed bias, MAE, RMSE, percentiles, cross-talk slope, repeatability across trials, tested range), with angle differences wrapped across +-180
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k metrics -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=7b2a9bdfcfe2812a31c3fb30d826f47d150b13a0947565a4af4a44f445eadb80; exit=0; EXPECT=matched; output-sha256=3465abfb3784746fd682d880289ae438f4589862b8b523badec62048dcc27e27; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V7: samples in tracking gaps, identity changes, invalid-status cuts, outside every take, in a take of another movement, or missing a reference coordinate are excluded with their reason and counted; the finger CMC of this tracker is reported unobservable, not fitted
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k gaps -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=318d6d0348053db46bd98b852514f42622fb76200caed672ec5de558a15d724b; exit=0; EXPECT=matched; output-sha256=49b2f1f1bfa104f1ade4ecedf49f5b5423cedeec004b0689e51bbae9a1a426cb; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V8: calibration and held-out samples never overlap: the mapping is fitted on the calibration trials only, the metrics use the held-out trials only, and a split that leaves no held-out trial, or no calibration trial, is refused
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_reference_validation.py -k split -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=ac391450428a42dace077b5768ddccd8d46317b68205cba0ac3c42e7af5d6dec; exit=0; EXPECT=matched; output-sha256=f547aec0357dc1248202b23852c4979ec2bae34c5c2d23b1b83b38bd21551f6e; output-bytes=115; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V9: the measurement status reaches the viewer, the per-frame export, its checker and the coverage table: tracker coordinate, unvalidated estimate, reference-evaluated (with its scope) and unavailable, the mapped cells empty outside the mapping's scope and tested range
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_angle_coverage.py tests/test_replay_viewer.py tests/test_replay_verify.py -k "mapping or status or coverage" -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=ee6738f1c3a3fa27ecfe3b14e0f5674159dab0867f6df438d92540c8102a038d; exit=0; EXPECT=matched; output-sha256=f83902c68367330f083b3cec4ef0913cd91ee596035e22aa6bb005a32b6440d2; output-bytes=116; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V10: the end-to-end example runs through the command line from a synthetic tracker session and a synthetic reference capture to a report and a mapping that both say synthetic, never evaluated
  CHECK: .venv\Scripts\python.exe scripts\leap\reference_validation.py example --out results\replay_fidelity\reference_validation_20261024\example
  EXPECT: /EXAMPLE DONE: data_origin=synthetic status=synthetic_demonstration[\s\S]*heldout_samples=[1-9]/
  EVIDENCE: automatic-evidence=v1; definition-sha256=e8fd94aba4fac107fd4de7d5183cd34cf0480ff36a7fa4f1c442ea191e7441b0; exit=0; EXPECT=matched; output-sha256=c29d20fdfa3b73b4ec080c10ea1d332ad2d60535143c1cdb9bb6aa345e94478e; output-bytes=875; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V11: on the real dataset, every selected frame is measured for whether the tracker's metacarpals move against its palm: the four finger metacarpals do not (the finger CMC is unobservable by this tracker), the thumb metacarpal does
  CHECK: .venv\Scripts\python.exe scripts\leap\reference_validation.py observability --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized" --out results\replay_fidelity\reference_validation_20261024\observability
  EXPECT: /OBSERVABILITY: recordings=167 frames=71709 [\s\S]*VERDICT: finger_cmc=unobservable thumb_tmc=observable/
  EVIDENCE: automatic-evidence=v1; definition-sha256=2579c60426938e398e5ca99ab59ce6cdb36ce382682a1ba711de434620f29996; exit=0; EXPECT=matched; output-sha256=8303c3f6d62525fb874fe935e216f3b8715182abb9c52c918b9a698265452614; output-bytes=416; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V12: the capture session runs end to end on the mock (every protocol item, sidecars, manifest with hashes) and its files load into the comparison's tracker-session reader
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_record_validation.py -q -p no:cacheprovider
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=38b8e9db19a0a8da5d2ddf4ade7b86d6b2517f59333382c9a3fe0ecacb97f588; exit=0; EXPECT=matched; output-sha256=5987bcb904938fa39248ee2602859887be59c3a74f14ef9713f97a37a9251de1; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V13: the whole dataset still reconstructs within the declared tolerances, the angle exports carry the estimate and the mapped columns, and the example's mapping is applied to no dataset recording (its scope is another participant's session)
  CHECK: .venv\Scripts\python.exe scripts\leap\replay_verify.py --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized" --out results\replay_fidelity\reference_validation_20261024\batch --all --scenarios --check-angle-exports --tmc-mapping results\replay_fidelity\reference_validation_20261024\example\comparison\mapping.json
  EXPECT: /FIDELITY OK: static=115 transitions=52 frames=71709 within_tolerance=167\/167[\s\S]*ANGLE EXPORTS OK: rows=71709 mcp_columns=16 est_columns=4 unavailable_empty=6 est_filled=71709 mapped_columns=2 mapped_filled=0/
  EVIDENCE: automatic-evidence=v1; definition-sha256=544ad75f943b995b3f6da8692d4ab821e09af49b92dbeef0d5cd82f7805e48bf; exit=0; EXPECT=matched; output-sha256=cd8493e8036b0715db084235e325e485135447ffdcf4021f3aed77915b5716af; output-bytes=1405; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V14: the four scenario reports of that run hold the known facts
  CHECK: .venv\Scripts\python.exe scripts\leap\replay_verify.py --check-scenarios results\replay_fidelity\reference_validation_20261024\batch
  EXPECT: SCENARIOS OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=ab842f018fbd9a639d2831b63dccd596b28df5268fa80eb85d6d88ce383274a9; exit=0; EXPECT=matched; output-sha256=a26400109739b9eab637baab4b3cc0bf24af033c7d2c43fc05d7199a7984406c; output-bytes=14; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V15: the dataset is unchanged: every file of the forensic baseline hashes to its baseline SHA-256
  CHECK: .venv\Scripts\python.exe scripts\leap\replay_verify.py --check-dataset-unchanged "C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\baseline_sha256.json" "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized"
  EXPECT: DATASET UNCHANGED
  EVIDENCE: automatic-evidence=v1; definition-sha256=7759d2ddcdacea4fe89e5943974205eccc4f2ac759fc1fcdd0fa5abf2afb0744; exit=0; EXPECT=matched; output-sha256=d38204bbde96ec4916ca82f85f1b0410afb5c8aed343720ebd01558ce2ad3073; output-bytes=189; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V16: the focused replay, calibration and validation tests pass together
  CHECK: .venv\Scripts\python.exe -B -m pytest tests/test_replay_core.py tests/test_paper3.py tests/test_replay_viewer.py tests/test_replay_verify.py tests/test_angle_coverage.py tests/test_tmc_estimate.py tests/test_tmc_calibration_take.py tests/test_reference_validation.py tests/test_record_validation.py -q -p no:cacheprovider
  EXPECT: /\d+ passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=38f3a49834dfe48dcd536deb141b09afaaff1c34d1f40f5240b3cae4e2da3434; exit=0; EXPECT=matched; output-sha256=bea7b5aa076797a6e62201949143771e3a372b4d5f34c39ab2c649fe1d02a7e4; output-bytes=438; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V17: the whole test suite passes
  CHECK: .venv\Scripts\python.exe -m pytest tests -q -x -p no:cacheprovider
  EXPECT: /\d+ passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=9ed50474bc76e66c2b56d3c7f1af8ac2458e6c8c43c766accbe7fcbbe0c84233; exit=0; EXPECT=matched; output-sha256=ac4368194bb05e0ffbac1465f1858747662b090aaf8e87f9950d3fda95fa3cbf; output-bytes=1828; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V18: this ledger lints clean
  CHECK: node C:\Users\nkim2\.claude\skills\unlazy\scripts\gate-lint.mjs GATES_reference_validation.md
  EXPECT: LINT OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=15b9de31f863b5e00bb59b28cbd8eb44f98314ce42f79941160a8b4da363072a; exit=0; EXPECT=matched; output-sha256=06897a108c52e191be6d899b39ae8a9e861645146d05029a453127fd86e0fa8a; output-bytes=172; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] V19: the viewer, driven headless on the example's synthetic take and on a real dataset take with the example's mapping, shows the reference-mapped block labelled synthetic in the first and not applied, with the reason, in the second; the four statuses are legible on screen (screenshots looked at)
  EVIDENCE: 2026-10-24, looked at by eye after the example was regenerated. results\replay_fidelity\reference_validation_20261024\viewer\01_synthetic_take_mapped.png (example tmc_flexion_sweep take 2, frame 100, with the example mapping and its neutral calibration): row 'mapped (SYNTHETIC, not measured): T_TMC_fe_mapped +30.3 T_TMC_aa_mapped +9.6', below it 'tested fe -20..41, aa 6..41 deg; SYNTHETIC-P00, left hand, session ...', then the estimate block with '(unvalidated estimate)' rows; console 'applied to this recording (inside its tested range only)'. 02_dataset_take_not_applied.png (dataset p3_circular_power take 1, same mapping): 'mapped (SYNTHETIC, not measured): not applied to this recording' and 'this recording is of session 20261001_101621_left, the mapping of session ...'; no mapped value anywhere. Both: title 'Angles of the displayed source frame (degrees; calculated = tracker coordinate)', rows labelled calculated / outside Paper 3, the one-row unavailable line naming the six, nothing cut at the panel's bottom (panel crops *_panel_crop.png).

- [ ] H1: the thumb TMC pair measured against an independent anatomical reference on a real participant, with held-out bias, MAE, RMSE, cross-talk and repeatability

- [ ] H2: a live capture on the hardware with the reference device recording at the same time, its synchronization checks passed on real data

- [ ] H3: the four finger CMC measured by an independent segment sensor (each metacarpal against the carpus)

ABANDON: H1 needs a reference device (optical motion capture with ISB-defined trapezium and first-metacarpal frames, an electromagnetic sensor pair, or at least a goniometer and a second person), a participant session and its recordings; none exist on 2026-10-24. The software path, format and protocol are ready (docs/validation_capture_protocol.md); no synthetic run counts here.
ABANDON: H2 needs the Ultraleap camera, the reference device and a human operator at the same time; this session has neither the device nor the operator. Mock rehearsal only (V12).
ABANDON: H3 cannot be met by this tracker at any calibration: its palm is one rigid frame (V11 measures it on every frame of the dataset). It needs markers or sensors on each metacarpal and on the carpus, recorded by a reference device; none exists here.
