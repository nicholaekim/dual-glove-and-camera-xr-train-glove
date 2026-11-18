# Gates: human-readable report (HTML + PDF) of the camera-clock erratum

OWNS: scripts/leap/camera_clock_erratum_report.py, tests/test_camera_clock_erratum_report.py, scripts/leap/camera_clock_erratum.py, tests/test_camera_clock_erratum.py, GATES_erratum_report.md

Scope: `scripts/leap/camera_clock_erratum_report.py` turns an erratum folder (README.md, static_takes.csv, transitions.csv, summary.md, optional VERIFICATION.md) into a paper-style `report.html` (self-contained, figures as base64 PNG) and `report.pdf` (headless Edge or Chrome), every number read from those files at run time; the outlier figure is drawn from the raw JSONL only with `--dataset` and is skipped, with the reason in the report, when the take cannot be loaded; the erratum README names the report; the dataset is never written.

- [x] R1: the report tests pass: title, the summary line's counts, every section heading, embedded figures, appendix row counts equal to the CSV row counts, the How-to-apply wording, the VERIFICATION content when present, a PDF starting with %PDF when a browser exists, the outlier figure drawn when its take loads and skipped (exit 0, reason in the report) when it does not
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_camera_clock_erratum_report.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=89abe5da01ed57720f5a905eca9ba97c861b21a397e5b201fca0473c645825ef; exit=0; EXPECT=matched; output-sha256=5a3f991b9f5a6f29844df76992f22b3772c2579f1bbd3cbc7ac0c6cbe5b992ce; output-bytes=101; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] R2: the erratum tests still pass, the README phrase naming the report pinned among them
  CHECK: .venv\Scripts\python.exe -m pytest tests/test_camera_clock_erratum.py -q
  EXPECT: / passed/
  EVIDENCE: automatic-evidence=v1; definition-sha256=4e6c5c0679cfac3a49f64a67322f96d3181608c4fd9b45e527a0deb895fa54a4; exit=0; EXPECT=matched; output-sha256=1fcdfbbaae62e4c85b634bed7371763b460312b673b18720ec47b020be74a96c; output-bytes=100; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] R3: the real run writes report.html and report.pdf beside the hand-in erratum tables, with the outlier figure drawn
  CHECK: .venv\Scripts\python.exe scripts\leap\camera_clock_erratum_report.py --erratum "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 camera-clock erratum" --dataset "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized"
  EXPECT: /REPORT OK: static=115 transitions=52 over_one_period=62 figures=4 pdf=yes/
  EVIDENCE: automatic-evidence=v1; definition-sha256=d36246c291ac27d43ae4a78288fc516fce79a9723efe1f47392cf1cdcc4ef053; exit=0; EXPECT=matched; output-sha256=10c5c31954bcb3e43032c064f566f819e78abbb7e166f836dceaa6e366adf01b; output-bytes=295; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries

- [x] R4: every page of the real report.pdf was looked at: nothing cut off or overlapping, no orphaned heading, no table split mid-row, figure labels readable, numbers agree with the CSVs
  EVIDENCE: 2026-10-29, final report.pdf (12 pages, A4) rendered page by page with pypdfium2 and read; earlier runs fixed: Appendix B wider than the page (whole document shrunk to fit), Table 2 caption and header orphaned under Figure 2, Figure 1 labels over the tallest bar, break marks touching tick labels. Numbers: a separate script (Decimal arithmetic, its own HTML parsing) compared every cell of Table 2, Appendix A and B, Tables 1/3/4 and the abstract's figures with static_takes.csv, transitions.csv and summary.md: NUMBERS AGREE; the same script flags a tampered copy (positive control).

- [x] R5: the dataset is unchanged: every file of the forensic baseline hashes to its baseline SHA-256
  CHECK: .venv\Scripts\python.exe scripts\leap\replay_verify.py --check-dataset-unchanged "C:\Users\nkim2\AppData\Local\Temp\hand-forensic-20261023-iqxo3s8o\baseline_sha256.json" "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set 2026-10-08 organized"
  EXPECT: DATASET UNCHANGED
  EVIDENCE: automatic-evidence=v1; definition-sha256=7759d2ddcdacea4fe89e5943974205eccc4f2ac759fc1fcdd0fa5abf2afb0744; exit=0; EXPECT=matched; output-sha256=d38204bbde96ec4916ca82f85f1b0410afb5c8aed343720ebd01558ce2ad3073; output-bytes=189; shell=C:\Windows\system32\cmd.exe; cwd=C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer; path=26dc5f0e5c3e/38 entries
