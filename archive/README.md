# archive/

Frozen snapshots of earlier repositories, merged in with their full git
history (`git subtree add`). Nothing here is maintained; the live code is at
the repository root.

## summer-xr-trainer (July 2026)

The original glove-only pipeline, merged on 2026-09-16 from
https://github.com/nicholaekim/summer-xr-trainer .

Unique material that exists only here:

- `recordings/poses/` : the July dataset, 42 glove takes of 7 poses on both
  hands, with `README.txt`, `REPORT.txt` (38/42 leave-one-out, core five 30/30),
  and `keypoints_txt/` (professor-format exports).
- `XR_Trainer_Technical_Documentation.pdf` and `README.md` : the July
  write-up of the OSC stream, data model and results.
- `scripts/record_frame.ps1`, `scripts/record_and_export.ps1`,
  `scripts/make_docs.py` : the professor-frame replication wrapper and the
  doc builder, never copied to the live tree.

`src/xr_hand`, the Python scripts and the tests are byte-identical to the
copies that were consolidated into the root `src/xr_hand` and
`scripts/glove/` in August 2026; use those.
