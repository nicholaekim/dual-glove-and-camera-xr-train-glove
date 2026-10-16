"""Check a built grasp hand-in folder (no hardware).

  .venv\\Scripts\\python.exe scripts\\validate_grasp_set.py "..\\xr trainer\\grasp set for professor 2026-10-01"
  .venv\\Scripts\\python.exe scripts\\validate_grasp_set.py <folder> --session recordings\\protocol\\grasps\\<session> --write

Reads the finished folder back and measures everything again from the files
in it (`leap_hand.grasp_validate`): the grasp types and their recordings
against the target, every duration from the timestamps, the other hand, the
hand's identity, every keypoints, all-frames and joint-frame file against
the recording it came from, the angles, the transition table, the README's
sections and counts. Prints the report, which ends with four lines:

  STATIC-POSE TASK: COMPLETE | INCOMPLETE          the task as it was set
  FULL MULTI-TAKE DATA COLLECTION: COMPLETE | INCOMPLETE
                                         our replication target
  TRANSITION TIMING: PARTIAL | MISSING   a valid timing of a recorded
                                         change of pose, or none
  FOLDER CHECK: PASSED | FAILED (n error(s))       the folder itself

`--session` also checks session.json's count of accepted takes against the
folder's. `--write` puts the report in the folder as validation_report.txt
(without the time of the check, so an unchanged folder gives the same file).
A folder whose transition table has a row that says MOCK (a take of a mock
transition session, a synthetic hand, added by a build with --allow-mock)
fails unless `--allow-mock` is given here too: a rehearsal, never the
hand-in.
Exit code 0 when the folder check passed, 1 when it failed, 2 when the
folder is not there. An incomplete collection is not a failure.
"""
import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leap_hand import grasp_handin as gh
from leap_hand import grasp_validate as gv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Check a built grasp hand-in folder against its own "
                    "files.")
    p.add_argument("folder", type=Path, help="the hand-in folder")
    p.add_argument("--session", type=Path, default=None,
                   help="the Set A session folder it was built from")
    p.add_argument("--write", action="store_true",
                   help=f"write the report into the folder as "
                        f"{gh.VALIDATION_FILE}")
    p.add_argument("--allow-mock", action="store_true",
                   help="pass a folder whose transition rows say MOCK (built "
                        "with --allow-mock); only for rehearsing, never for "
                        "the hand-in")
    args = p.parse_args(argv)
    if not gh.long_path(args.folder).is_dir():
        print(f"not a folder: {args.folder}")
        return 2
    report = gv.validate(args.folder, args.session,
                         allow_mock=args.allow_mock)
    print(gv.render(report, time.strftime("%Y-%m-%d %H:%M")), end="")
    if args.write:
        (gh.long_path(args.folder) / gh.VALIDATION_FILE).write_text(
            gv.render(report), encoding="utf-8")
        print(f"wrote {args.folder / gh.VALIDATION_FILE}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
