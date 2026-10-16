"""Re-time the change into the grasp of a recorded Set A session, in place.

The rule that times it (`leap_hand.transition`) changed on 2026-10-02, after
the first forming files: the settle is now searched for after the hand's
main movement, its fastest frame, and a take whose hand never moved at
MOVE_SPEED_MM_S after the cue has no transition and a note instead. The
takes recorded before that carry the first rule's `transition_s`. This
reads every take's forming file again and rewrites only the meta's
`coaching` keys that come from it:

  transition_s, peak_speed_mm_s, peak_at_s, last_move_s, transition_note,
  transition_rule

and nothing else: not the forming file, not session.json, not another key
of the meta. `forming_frames`, `forming_tracked_fraction` and `form_start`
stay as the recorder wrote them.

  python scripts/leap/recompute_transition.py recordings/protocol/grasps/<session> --dry-run
  python scripts/leap/recompute_transition.py recordings/protocol/grasps/<session>

Every meta with a forming file is redone, kept takes (`meta/`) and attempts
not kept (`rejected/meta/`). The hand measured is the one the recorder
measured: the last open hand OPEN HAND acquired (`coaching.acquired`),
followed by id beside the other hands the coaching saw
(`coaching.other_hands`), on its lines labelled as the operator's hand
(`hand`), from the cue (`coaching.form_start`) to the start of the hold
(`take_start`). A meta is written back exactly as the recorder writes one
(`json.dumps(..., indent=2)`, the file's own line endings), and only after
checking that the file on disk is in that form already and that the result
differs from it in those keys only; one that fails either check is left
alone and named. One line per take, old -> new, then `recompute dry-run ok`
or `recompute ok` (exit 0), or how many were not rewritten (exit 1).
"""
import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leap_hand.transition import summarise_forming

# The meta's `coaching` keys this rewrites, in the order the recorder writes
# them; transition_rule stays where it is.
KEYS = ("transition_s", "peak_speed_mm_s", "peak_at_s", "last_move_s",
        "transition_note", "transition_rule")
META_DIRS = ("meta", "rejected/meta")


def render(data: dict, newline: str = "\n") -> bytes:
    """A meta's bytes as the recorder's `write_json` puts them on disk."""
    return (json.dumps(data, indent=2) + "\n").replace("\n", newline).encode("utf-8")


def file_newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def read_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def without_keys(meta: dict) -> dict:
    """The meta with the rewritten keys taken out of `coaching`."""
    out = dict(meta)
    coaching = out.get("coaching")
    if isinstance(coaching, dict):
        out["coaching"] = {k: v for k, v in coaching.items() if k not in KEYS}
    return out


def with_numbers(coaching: dict, numbers: dict) -> dict:
    """`coaching` with `numbers` in it: each key where the recorder puts it
    (the new ones right after transition_s), every other key untouched."""
    out = {}
    for key, value in coaching.items():
        if key in KEYS and key not in ("transition_s", "transition_rule"):
            continue
        if key == "transition_s":
            for k in KEYS[:-1]:
                out[k] = numbers[k]
            continue
        out[key] = numbers[key] if key == "transition_rule" else value
    for k in KEYS:
        out.setdefault(k, numbers[k])
    return out


def forming_numbers(session: Path, meta: dict) -> dict:
    """`summarise_forming` on the take's forming file, as the recorder ran it:
    the meta's `coaching` transition keys (`FormingSummary.coaching_numbers`)."""
    coaching = meta["coaching"]
    rows = read_rows(session / meta["files"]["forming"])
    acquired = coaching.get("acquired") or []
    follow = None
    if acquired and acquired[-1].get("hand_id") is not None:
        follow = (int(acquired[-1]["hand_id"]),
                  [int(i) for i in coaching.get("other_hands") or []], None)
    t0 = float(coaching["form_start"])
    t_end = meta.get("take_start")
    if t_end is None:
        t_end = max(float(r.get("capture_time") or r["wall_time"]) for r in rows)
    hand = meta.get("hand")
    s = summarise_forming(rows, t0, float(t_end), follow=follow,
                          require_label=hand, prefer=hand)
    return s.coaching_numbers()


def _num(v, unit: str = "") -> str:
    return "null" if v is None else f"{v:g}{unit}"


def recompute(session: Path, meta_path: Path
              ) -> Tuple[str, Optional[bytes], Optional[str]]:
    """(the take's line, the meta's new bytes or None when unchanged, why it
    is not rewritten or None)."""
    name = meta_path.stem
    raw = meta_path.read_bytes()
    nl = file_newline(raw)
    meta = json.loads(raw.decode("utf-8"))
    if render(meta, nl) != raw:
        return f"  {name}: not rewritten", None, (
            "the file is not in the recorder's json form; left as it is")
    coaching = meta.get("coaching")
    if not isinstance(coaching, dict) or coaching.get("form_start") is None:
        return f"  {name}: not rewritten", None, "no coaching.form_start in the meta"
    if not (session / meta["files"]["forming"]).is_file():
        return f"  {name}: not rewritten", None, (
            f"forming file {meta['files']['forming']} is missing")
    numbers = forming_numbers(session, meta)
    new = dict(meta)
    new["coaching"] = with_numbers(coaching, numbers)
    out = render(new, nl)
    # Only those keys may differ: the rest re-rendered must be the same bytes.
    if render(without_keys(new), nl) != render(without_keys(meta), nl):
        return f"  {name}: not rewritten", None, "would change more than the transition keys"
    old_t, new_t = coaching.get("transition_s"), numbers["transition_s"]
    line = f"  {name}: transition_s {_num(old_t)} -> {_num(new_t)}"
    line += (f"   (peak {_num(numbers['peak_speed_mm_s'])} mm/s at "
             f"{_num(numbers['peak_at_s'])} s, last move "
             f"{_num(numbers['last_move_s'], ' s')})")
    if numbers["transition_note"]:
        line += f"   note: {numbers['transition_note']}"
    if out == raw:
        return line + "   (unchanged)", None, None
    changed = sum(1 for d in difflib.ndiff(raw.decode("utf-8").splitlines(),
                                           out.decode("utf-8").splitlines())
                  if d[:1] in "+-")
    return line + f"   ({changed} lines of the meta differ)", out, None


def write_bytes(path: Path, data: bytes) -> None:
    """Through a temporary file, retried briefly as the recorder's
    `write_json` does: OneDrive can hold a file for a moment."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def forming_metas(session: Path) -> List[Path]:
    """Every meta of the session with a forming file named in it."""
    found = []
    for sub in META_DIRS:
        for p in sorted((session / sub).glob("*.json")):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict) and (meta.get("files") or {}).get("forming"):
                found.append(p)
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Rewrite the transition keys of every take with a forming "
                    "file in one Set A session, with today's rule.")
    p.add_argument("session", type=Path, help="the session folder (session.json)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would change, write nothing")
    args = p.parse_args(argv)
    session = args.session
    if not (session / "session.json").is_file():
        print(f"no session.json in {session}")
        return 2
    metas = forming_metas(session)
    print(f"{session}: {len(metas)} take(s) with a forming file"
          + ("  (dry run: nothing is written)" if args.dry_run else ""))
    failed = 0
    for path in metas:
        line, new, why = recompute(session, path)
        if why is not None:
            failed += 1
            print(f"{line}: {why}")
            continue
        print(line)
        if new is not None and not args.dry_run:
            write_bytes(path, new)
            if path.read_bytes() != new:
                failed += 1
                print(f"  {path.stem}: written, but the file does not read back")
    mode = "recompute dry-run" if args.dry_run else "recompute"
    if failed:
        print(f"{mode}: {failed} take(s) not rewritten")
        return 1
    print(f"{mode} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
