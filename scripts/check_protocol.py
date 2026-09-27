"""Pass or fail every take of one protocol session, while the gloves are still on.

The plan's section 6 says "check_protocol.py after each set and hand, redo
failed takes before taking the gloves off". This is that command: point it at
one session folder written by `scripts/record_protocol.py` (Sets B and C) or
`scripts/leap/record_poses.py --protocol` (Set A) and it prints a table, one
row per take, accepted and rejected, then the diagnostics per item, then the
rules it applied, and writes the same as `check.csv` and `check.txt`.

  python scripts/check_protocol.py recordings/protocol/finger_flexion/20260928_140501_left
  python scripts/check_protocol.py recordings/protocol/sequences/20260928_143012_right --bands auto
  python scripts/check_protocol.py <session> --out results/protocol_check

What it checks is in `cam_hand.protocol_check` (and the rules are printed at
the bottom of every report): the cued finger's span and the cycle counts for
Set B, every step's glove and camera fractions for Set C, the acquisition
gate for Set A. The fixed numbers are acquisition gates; the continuous
values behind each verdict are always in the output, so a take can be
re-judged from the CSV without re-running this.

`--bands auto` is for Set C after the first hand: the flexed and straight
bands are re-derived from this session's own open-hand and full-fist steps
instead of the warm-up (plan section 4).

Exit code 1 when an accepted take fails a rule, 0 otherwise (2 when the
folder holds no session.json). A rejected attempt is checked and listed but
never changes the exit code: the recorder already threw it out. It runs on
a glove-only session (no `leap/`) and on a session without `stills/`.
"""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cam_hand.protocol_check import BANDS, FIXED, check_session, write_check


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Check every take of one protocol session (Sets A, B "
                    "and C) and write check.csv and check.txt.")
    p.add_argument("session", type=Path,
                   help="session folder holding session.json (contract "
                        "docs/protocol_formats.md section 1)")
    p.add_argument("--bands", choices=BANDS, default=FIXED,
                   help="Set C bands: fixed = flexed > 0.6, straight < 0.3 of "
                        "the warm-up range; auto = the same shares of this "
                        "session's own open-hand to full-fist range "
                        "(default: fixed)")
    p.add_argument("--out", type=Path, default=None,
                   help="folder for check.csv and check.txt (default: the "
                        "session folder)")
    args = p.parse_args(argv)

    if not (args.session / "session.json").is_file():
        print(f"no session.json in {args.session}\n"
              "  point this at one session folder, e.g. "
              "recordings/protocol/finger_flexion/<stamp>_<hand>")
        return 2
    check = check_session(args.session, bands=args.bands)
    csv_path, txt_path, text = write_check(check, args.out)
    print(text)
    print(f"wrote {csv_path}")
    print(f"wrote {txt_path}")
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
