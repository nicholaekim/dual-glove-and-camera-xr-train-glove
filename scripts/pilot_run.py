"""One command per step of the hand-on-object pilot (docs/pilot_hand_on_object.md).

    .venv\\Scripts\\python.exe scripts\\pilot_run.py A --gloves left   step 4; --gloves once, the first time
    .venv\\Scripts\\python.exe scripts\\pilot_run.py B1                steps 5 to 8: B1, C, B2, A2
    .venv\\Scripts\\python.exe scripts\\pilot_run.py compare           step 9, once A2 is complete
    .venv\\Scripts\\python.exe scripts\\pilot_run.py close --reason "<why>"   end a pilot that cannot be finished
    .venv\\Scripts\\python.exe scripts\\pilot_run.py C --print         show the commands, record nothing

A run records its three grasps one at a time, each with its exact
scripts/record_simultaneous.py command: one grasp (--poses), only the takes
it still lacks (--takes), numbered on from the kept ones (--first-take),
--keep-all-tracking for the object runs B1, C and B2 (the amendment of
2026-10-23: every attempt whose hold ran is kept, the camera's losses logged
in its meta), the hint, the out-dir, and a --note naming the grasp's object
and the glove setup. So no command is ever edited by hand, and a run that was cut short is
topped up by running the same step again.

Before anything is recorded it checks, in this order: the glove setup, one
for the whole pilot, kept in <root>\\glove_setup.txt (written by --gloves
left|both); for the object runs B1, C and B2, <root>\\objects.txt with one
valid line per object, frozen in <root>\\objects.lock when B1 starts; the
order (the previous run COMPLETE, every grasp with its expected accepted
takes; no later run STARTED, i.e. no attempt, kept or rejected, in its folder
and no line in <root>\\runs_started.txt); and that the run is not complete
already. Every unmet check is printed, one paragraph each in that order, with
exit code 2, and nothing is written or run.

`compare` runs scripts/pilot_compare.py once A2 is complete, or after the
pilot was closed as incomplete with `close --reason`. pilot_compare.py
--root itself stays usable directly at any time, for diagnostics.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = r"recordings\pilot_hoo"
GRASPS = ("cylindrical_object", "spherical_object", "tip_object")
OBJECT_IDS = ("OBJ-CYL", "OBJ-BALL", "OBJ-PEN")      # one per grasp, in order
GRASP_OBJECT = dict(zip(GRASPS, OBJECT_IDS))
GLOVE_TEXT = {"left": "left glove only", "both": "both gloves connected"}
_OBJECTS = "OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt"

# the protocol's order: run, folder, takes per grasp, hint, condition, note
# (the --note before "; grasp ...; <glove text>")
RUNS = (
    ("A", "A_mimed", 5, None, "Condition A, mimed",
     "condition A mimed, no object"),
    ("B1", "B1_object", 3, None, "Condition B1, object",
     f"condition B1 object: {_OBJECTS}"),
    ("C", "C_object_hint", 5, "hand_on_object",
     "Condition C, object with the hint",
     f"condition C object with hand_on_object: {_OBJECTS}"),
    ("B2", "B2_object", 2, None, "Condition B2, object",
     f"condition B2 object after C: {_OBJECTS}"),
    ("A2", "A_mimed_again", 1, None, "Drift check, condition A again",
     "condition A repeated at the end, drift check"),
)
RUN = {r[0]: r for r in RUNS}
LABELS = tuple(RUN)
OBJECT_RUNS = ("B1", "C", "B2")

MIMED = ("Nothing in the hand: open hand steady over the module, beep, form "
         "the grasp in the air, hold still for the 5 s, open.")
HANDLING = ("Pick the object up with the right hand and place it into the "
            "open left hand after the tracker has acquired the open hand, "
            "close on it at the beep, hold still for the 5 s, put it down "
            "between takes. Hold, do not squeeze.")
HINTED = ("The camera is asked for hand_on_object; the recorder confirms it "
          "before recording.")


class Refusal(Exception):
    """A precondition not met: printed as one paragraph, exit code 2."""


def shell(argv: list) -> str:
    """argv as the one line typed at the repo root."""
    words = [r".venv\Scripts\python.exe"]
    words += [f'"{w}"' if " " in w else w for w in argv[1:]]
    return " ".join(words)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --- the glove setup and the objects -----------------------------------------
def glove_setup(root_dir: Path, root: Path, asked):
    """(setup, new): the pilot's glove setup, and whether the file is still
    to be written."""
    path, shown = root_dir / "glove_setup.txt", root / "glove_setup.txt"
    if not path.is_file():
        if asked is None:
            raise Refusal(
                "no glove setup yet: run this step with --gloves left (the "
                "left glove only) or --gloves both (both gloves connected) "
                f"once; it is kept in {shown} for every later run.")
        return asked, True
    kept = path.read_text(encoding="utf-8-sig").strip()
    if kept not in GLOVE_TEXT:
        raise Refusal(
            f'{shown} says "{kept}", not left or both; it holds the glove '
            "setup of this pilot's runs, so put back the word they were "
            "recorded with.")
    if asked is not None and asked != kept:
        raise Refusal(
            "the pilot's five runs use one glove setup; glove_setup.txt says "
            f'"{kept}" and you asked for "{asked}". Record the whole pilot '
            "again in a new --root if the setup changed.")
    return kept, False


def positive(text: str) -> bool:
    try:
        x = float(text)
    except ValueError:
        return False
    return math.isfinite(x) and x > 0


def read_objects(root_dir: Path, root: Path) -> list:
    """The OBJ-CYL, OBJ-BALL and OBJ-PEN lines of objects.txt, as written."""
    path, shown = root_dir / "objects.txt", root / "objects.txt"
    form = "| what it is | diameter_mm | length_mm | mass_g"
    if not path.is_file():
        raise Refusal(
            f'no {shown} yet: one line per object, "OBJ-CYL {form}", mass '
            'may be "not measured"')
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise Refusal(f"{shown} is not UTF-8 text: save it again as UTF-8."
                      ) from None
    lines = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split("|")]
        name = fields[0]
        if name not in OBJECT_IDS:
            raise Refusal(
                f'{shown}: the line "{line}" is not for one of '
                f"{', '.join(OBJECT_IDS)}; the pilot uses these three objects "
                "and no others.")
        if name in lines:
            raise Refusal(f"{shown} has two lines for {name}; keep one.")
        if len(fields) != 5:
            raise Refusal(
                f'{shown}: the {name} line has {len(fields)} fields, not 5: '
                f'"{line}"; the form is "{name} {form}".')
        for what, value in (("diameter", fields[2]), ("length", fields[3])):
            if not positive(value):
                raise Refusal(f'{shown}: the {name} {what} "{value}" is not '
                              "a positive number of mm.")
        if fields[4] != "not measured" and not positive(fields[4]):
            raise Refusal(f'{shown}: the {name} mass "{fields[4]}" is not a '
                          'positive number of g or "not measured".')
        lines[name] = line
    missing = [i for i in OBJECT_IDS if i not in lines]
    if missing:
        raise Refusal(f"{shown} has no line for {', '.join(missing)}: one "
                      f'line per object, "<ID> {form}".')
    return [lines[i] for i in OBJECT_IDS]


def lock_text(objects: list) -> str:
    """objects.lock: the sha256 of the normalized lines, then the lines."""
    body = "\n".join(" | ".join(f.strip() for f in line.split("|"))
                     for line in objects)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"sha256 {digest}\n{body}\n"


def check_lock(root_dir: Path, root: Path, objects: list) -> None:
    path = root_dir / "objects.lock"
    if not path.is_file():
        raise Refusal(
            f"no {root / 'objects.lock'}: B1 writes it when it starts, and C "
            "and B2 are checked against it, so the objects of B1 cannot be "
            "confirmed; put it back, or record the whole pilot again in a new "
            "--root")
    kept = path.read_text(encoding="utf-8-sig").splitlines()[:1]
    if kept != lock_text(objects).splitlines()[:1]:
        raise Refusal(
            f"{root / 'objects.txt'} changed since B1 started (objects.lock): "
            "the pilot's three objects are fixed; put the lines back, or "
            "record the whole pilot again in a new --root")


# --- what the run folders hold -----------------------------------------------
def kept_takes(run_dir: Path) -> dict:
    """{grasp: take numbers of its accepted takes} of one run folder."""
    kept = {g: [] for g in GRASPS}
    for path in (run_dir / "leap").glob("*.meta.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(meta, dict) and meta.get("accepted") is True
                and meta.get("pose") in kept):
            take = meta.get("take")
            kept[meta["pose"]].append(take if isinstance(take, int) else 0)
    return kept


def is_complete(kept: dict, n: int) -> bool:
    return all(len(kept[g]) >= n for g in GRASPS)


def progress(kept: dict, n: int) -> str:
    return ", ".join(f"{g} {len(kept[g])}/{n}" for g in GRASPS)


def incomplete(kept: dict, n: int) -> str:
    counts = ", ".join(f"{g} {len(kept[g])}" for g in GRASPS)
    return f"incomplete ({counts}, needs {n} per grasp)"


def has_started(root_dir: Path, label: str) -> bool:
    run_dir = root_dir / RUN[label][1]
    if any((run_dir / "leap").glob("*.meta.json")) or \
            any((run_dir / "rejected" / "leap").glob("*.meta.json")):
        return True
    path = root_dir / "runs_started.txt"
    return path.is_file() and any(
        line.split()[0] == label
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip())


def check_predecessor(root_dir: Path, root: Path, run: str) -> None:
    i = LABELS.index(run)
    if i == 0:
        return
    before, folder, n = RUNS[i - 1][:3]
    kept = kept_takes(root_dir / folder)
    if not is_complete(kept, n):
        raise Refusal(f"run {run} comes after {before}; {root / folder} is "
                      f"{incomplete(kept, n)}")


def check_later(root_dir: Path, run: str) -> None:
    for later in LABELS[LABELS.index(run) + 1:]:
        if has_started(root_dir, later):
            raise Refusal(
                f"run {run} cannot be recorded after {later} has started: "
                f"the protocol's order is {', '.join(LABELS)}")


def run_command(cmd: list) -> int:
    sys.stdout.flush()
    try:
        return subprocess.run(cmd, cwd=REPO).returncode
    except KeyboardInterrupt:
        # The recorder handles its own Ctrl+C (an interrupted take is deleted,
        # kept takes stay); this only keeps a traceback off the screen.
        print("\nstopped with Ctrl+C; the takes kept so far stay in place, "
              "and the same step can be run again")
        return 130


# --- step 9 and the close ------------------------------------------------------
def compare_or_close(a, root: Path, root_dir: Path) -> int:
    _, a2_folder, a2_n = RUN["A2"][:3]
    a2 = kept_takes(root_dir / a2_folder)
    closed = root_dir / "pilot_closed.txt"
    if a.run == "compare":
        if not is_complete(a2, a2_n) and not closed.is_file():
            again = "" if str(root) == DEFAULT_ROOT else f" --root {root}"
            print(f"the pilot is not finished: A2 is {incomplete(a2, a2_n)}; "
                  "run the remaining steps, or close it as incomplete with: "
                  r'.venv\Scripts\python.exe scripts\pilot_run.py close '
                  f'--reason "<why>"{again}')
            return 2
        cmd = [sys.executable, r"scripts\pilot_compare.py",
               "--root", str(root)]
        print(shell(cmd))
        if a.print_only:
            return 0
        print()
        print("No hardware. The comparison of the five runs: takes.csv, "
              f"summary.csv and report.txt go into {root / 'comparison'}.")
        return run_command(cmd)

    reason = (a.reason or "").strip()
    refusals = []
    if not reason:
        refusals.append('close needs --reason "<why>": why the pilot ends '
                        "incomplete is kept in pilot_closed.txt beside the "
                        "counts.")
    if is_complete(a2, a2_n):
        refusals.append("nothing to close: the pilot is complete")
    elif closed.is_file():
        refusals.append(f"the pilot is already closed: "
                        f"{root / 'pilot_closed.txt'} keeps the first time "
                        "and reason.")
    if refusals:
        print("\n\n".join(refusals))
        return 2
    rows = [f"pilot closed as incomplete {now()}"]
    rows += [f"{label} ({folder}): {progress(kept_takes(root_dir / folder), n)}"
             for label, folder, n, *_ in RUNS]
    rows.append(f"reason: {reason}")
    text = "\n".join(rows) + "\n"
    print(text, end="")
    if not a.print_only:
        root_dir.mkdir(parents=True, exist_ok=True)
        closed.write_text(text, encoding="utf-8")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run one step of the hand-on-object pilot "
                    "(docs/pilot_hand_on_object.md) with its exact commands.")
    p.add_argument("run", choices=LABELS + ("compare", "close"),
                   help="the run, in the protocol's order; compare (step 9); "
                        "or close, to end an unfinishable pilot")
    p.add_argument("--gloves", choices=tuple(GLOVE_TEXT),
                   help="the glove setup for the whole pilot, given once: "
                        "left (the left glove only) or both (both gloves "
                        "connected); kept in <root>\\glove_setup.txt")
    p.add_argument("--reason", help="close: why the pilot ends incomplete")
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="show the commands, record and mark nothing")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help=f"the pilot folder (default {DEFAULT_ROOT})")
    a = p.parse_args(argv)
    root = Path(a.root)
    root_dir = REPO / root           # the recorder runs from the repo root

    if a.run in ("compare", "close"):
        return compare_or_close(a, root, root_dir)

    _, folder, n, hint, condition, note = RUN[a.run]
    refusals = []

    def check(f, *args):
        try:
            return f(*args)
        except Refusal as e:
            refusals.append(str(e))

    lock = root_dir / "objects.lock"
    setup = check(glove_setup, root_dir, root, a.gloves)
    objects = (check(read_objects, root_dir, root) if a.run in OBJECT_RUNS
               else None)
    if objects is not None and (a.run != "B1" or lock.is_file()):
        check(check_lock, root_dir, root, objects)
    check(check_predecessor, root_dir, root, a.run)
    check(check_later, root_dir, a.run)
    kept = kept_takes(root_dir / folder)
    if is_complete(kept, n):
        refusals.append(f"run {a.run} is complete ({progress(kept, n)}); "
                        "nothing to record. Takes beyond the protocol are not "
                        "added.")
    if refusals:                     # every unmet precondition, in order
        print("\n\n".join(refusals))
        return 2
    gloves, new = setup
    if new:
        root_dir.mkdir(parents=True, exist_ok=True)
        (root_dir / "glove_setup.txt").write_text(gloves + "\n",
                                                  encoding="utf-8")

    # one invocation per grasp still short: the missing takes, numbered on
    plan = []
    for i, grasp in enumerate(GRASPS):
        missing = n - len(kept[grasp])
        if missing <= 0:
            continue
        first = max(kept[grasp], default=0) + 1
        middle = (f"grasp {grasp} with {objects[i]}" if objects
                  else f"grasp {grasp}, nothing in the hand")
        cmd = [sys.executable, r"scripts\record_simultaneous.py",
               "--camera", "leap", "--hand", "left", "--poses", grasp,
               "--takes", str(missing), "--first-take", str(first)]
        if a.run in OBJECT_RUNS:
            # protocol amendment 2026-10-23: with an object in the hand the
            # camera's losses are the result, so every completed hold is kept
            cmd += ["--keep-all-tracking"]
        if hint:
            cmd += ["--hint", hint]
        cmd += ["--out-dir", str(root / folder),
                "--note", f"{note}; {middle}; {GLOVE_TEXT[gloves]}"]
        plan.append((i, grasp, missing, first, cmd))
    for *_, cmd in plan:
        print(shell(cmd))
    if new:
        print(f"glove setup kept in {root / 'glove_setup.txt'}: "
              f"{GLOVE_TEXT[gloves]}, for every run of this pilot")
    if a.print_only:
        return 0

    total = len(GRASPS) * n
    print()
    print(f"Gloves on + camera. {condition}, {total} takes ({n} per grasp). "
          "Right hand on the table, outside the camera's view. Do not "
          "recalibrate the glove.")
    to_record = sum(p[2] for p in plan)
    if to_record < total:
        print(f"Top-up: {to_record} of the {total} takes still missing; only "
              "those are recorded, numbered on from the kept ones.")
    if objects is None:
        print(MIMED)
    else:
        print("Objects:")
        for line in objects:
            print(f"  {line}")
        print(HANDLING)
    if hint:
        print(HINTED)

    with open(root_dir / "runs_started.txt", "a", encoding="utf-8") as f:
        f.write(f"{a.run} {now()}\n")
    if a.run == "B1" and not lock.is_file():
        lock.write_text(lock_text(objects), encoding="utf-8")
    for i, grasp, missing, first, cmd in plan:
        print()
        print(f"NEXT GRASP: {grasp} ({missing} take{'s' * (missing != 1)}, "
              f"numbered from {first})")
        if objects:
            print(f"USE {GRASP_OBJECT[grasp]}: {objects[i]}")
        rc = run_command(cmd)
        if rc != 0:
            return rc
        # q in the recorder exits 0 with the grasp short: stop, do not skip it
        got = len(kept_takes(root_dir / folder)[grasp])
        if got < n:
            print(f"\n{grasp} is still short ({got}/{n}): stopped here so "
                  "the grasps stay in order; run this step again to top it "
                  "up.")
            return 3

    kept = kept_takes(root_dir / folder)
    print()
    print(f"run {a.run} now: {progress(kept, n)}; "
          + ("complete." if is_complete(kept, n) else
             "not complete: run this step again, it records only the "
             "missing takes."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
