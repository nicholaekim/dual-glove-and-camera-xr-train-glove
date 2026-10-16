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
the bottom of every report): the cued finger's span, the cycle counts and,
in a session recorded under the bend-speed rule, how long each bend took
for Set B (the times are reported for every session), every step's glove
and camera fractions for Set C, the acquisition gate for Set A. The fixed numbers are acquisition gates; the continuous
values behind each verdict are always in the output, so a take can be
re-judged from the CSV without re-running this.

`--bands auto` is for Set C after the first hand: the flexed and straight
bands are re-derived from this session's own open-hand and full-fist steps
instead of the warm-up (plan section 4).

Exit code 1 when an accepted take fails a rule, 0 otherwise (2 when the
folder holds no session.json). A rejected attempt is checked and listed but
never changes the exit code: the recorder already threw it out. It runs on
a glove-only session (no `leap/`) and on a session without `stills/`.

Set A takes recorded since 2026-10-02 also have a forming file (every frame
from the close cue to the take, `forming/<take>.jsonl`). Such a take must
say what was measured from it: `coaching.transition_s` present in its meta
(null is an answer, and `coaching.transition_note` says which: no close
after the cue, or never settled before the hold) and
`coaching.forming_frames` at least 1. The transition and its note go into
the take's notes, in check.csv and check.txt. A meta from before the
peak-anchored rule (no `peak_speed_mm_s`, `last_move_s`,
`transition_note`) passes with the note "recompute needed" and the command,
`scripts/leap/recompute_transition.py <session>`. A take without a forming
file, every take recorded before, is checked exactly as before
(`check_forming`).
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cam_hand.protocol_check import (BANDS, FAIL, FIXED, GRASPS, PASS,
                                     SessionCheck, TakeRef, check_session,
                                     read_json, write_check)


def forming_path(check: SessionCheck, ref: TakeRef, meta) -> Optional[Path]:
    """The take's forming file on disk, or None: session.json's
    `files.forming`, else the meta's."""
    found = ref.path("forming")
    if found is not None:
        return found
    files = meta.get("files") if isinstance(meta, dict) else None
    rel = files.get("forming") if isinstance(files, dict) else None
    p = check.session.path / str(rel) if rel else None
    return p if p is not None and p.is_file() else None


NEW_KEYS = ("peak_speed_mm_s", "last_move_s", "transition_note")


def forming_note(coaching: dict, frames: int) -> str:
    """The take's note on its transition: the number, or why there is none
    (`transition_note`, else the old reading of a null)."""
    t = coaching["transition_s"]
    why = coaching.get("transition_note") or ""
    if t is not None:
        return (f"transition {float(t):.2f} s, close cue to settled "
                f"({frames} forming frames)" + (f"; {why}" if why else ""))
    return (f"transition none: {why or 'the hand never settled before the take'}"
            f" ({frames} forming frames)")


def check_forming(check: SessionCheck) -> int:
    """Set A: the forming file's rule, on top of `check_session`'s.

    A take with a forming file fails unless its meta's `coaching` has the
    key `transition_s` (a number, or null with `transition_note` saying why:
    no close after the cue, or never settled before the hold) and
    `forming_frames` of at least 1; it then gets a note with the
    transition. A meta written before `peak_speed_mm_s`, `last_move_s` and
    `transition_note` existed is not failed for it: its note says
    "recompute needed" with the command that adds them. A take without a
    forming file is left as it was. The verdicts are recomputed, so the
    exit code and REDO line count the new failures. Returns how many takes
    had a forming file.
    """
    if check.session.set != GRASPS:
        return 0
    refs = {t.name: t for t in check.session.takes}
    seen = stale = 0
    for row in check.rows:
        ref = refs.get(row["name"])
        if ref is None:
            continue
        meta = read_json(ref.path("meta")) if ref.path("meta") else None
        if forming_path(check, ref, meta) is None:
            continue
        seen += 1
        coaching = meta.get("coaching") if isinstance(meta, dict) else None
        coaching = coaching if isinstance(coaching, dict) else {}
        failures = []
        if "transition_s" not in coaching:
            failures.append("forming file, but no coaching.transition_s in "
                            "the meta")
        frames = coaching.get("forming_frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            failures.append(f"forming file, but coaching.forming_frames is "
                            f"{frames!r} (need at least 1)")
        row["failures"] = list(row["failures"]) + failures
        if not failures:
            notes = [forming_note(coaching, frames)]
            missing = [k for k in NEW_KEYS if k not in coaching]
            if missing:
                stale += 1
                notes.append(
                    f"recompute needed: no coaching.{', '.join(missing)} "
                    "(scripts/leap/recompute_transition.py "
                    f"{check.session.path.as_posix()})")
            row["notes"] = list(row["notes"]) + notes
        row["verdict"] = FAIL if row["failures"] else PASS
    if seen:
        check.header.append(
            f"forming   {seen} take(s) with a forming file (close cue to the "
            "take); each needs coaching.transition_s (null allowed, the "
            "transition_note says why) and forming_frames >= 1 in its meta; "
            "the transition is in its notes")
    if stale:
        check.header.append(
            f"forming   {stale} take(s) timed by the first rule: recompute "
            "needed, not a failure (scripts/leap/recompute_transition.py "
            "<session>)")
    return seen


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
    check_forming(check)
    csv_path, txt_path, text = write_check(check, args.out)
    print(text)
    print(f"wrote {csv_path}")
    print(f"wrote {txt_path}")
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
