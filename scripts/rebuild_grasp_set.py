"""Rebuild the professor's grasp folder from a Set A session: one command.

  .venv\\Scripts\\python.exe scripts\\rebuild_grasp_set.py ^
      --session recordings\\protocol\\grasps\\20261001_101621_left ^
      --out "..\\xr trainer\\grasp set for professor 2026-10-01" --operator "N Kim"

  ... --transitions recordings\\protocol\\transitions\\<session>   adds the
      takes of a transition session (scripts\\leap\\record_transitions.py)
      to the transition table

The transition sessions are also found without being named.
`--transitions-dir` (default: the `transitions` folder beside the session's
own set folder, which for a session in recordings\\protocol\\grasps is
recordings\\protocol\\transitions) is looked through: every folder in it
whose session.json is one of the transition recorder's ("set":
"transitions") is added, in name order, after the folders given with
`--transitions`, and none twice. A mock session (session.json "mock": true,
recorded with --mock) is passed over unless `--allow-mock`. One line is
printed per folder taken or passed over, and one, "no transition sessions
in ...", when none is taken or the folder does not exist. So the same
command (rebuild_professor_grasp_set.cmd on the Desktop) picks up every
transition session recorded with the camera, and never a rehearsal, whose
sessions go to recordings\\protocol_mock\\transitions.

No hardware. Run it again after every recording session; it does, in order:

  1. JOINT FRAMES   `scripts/joint_frames_view.py --session`: for every
                    accepted take, the 26-row table of its representative
                    frame, and joint_frames.pdf, into the session's own
                    joint_frames folder (derived files; the recordings are
                    only read).
  2. THE FOLDER     `scripts/package_professor_set.py --grasps`: the
                    accepted recordings and their keypoints copied, each
                    grasp's frame-by-frame dataset, the summary, the grasp
                    audit, the 24 angles, the joint key, each take's
                    infrared still cut to the recorded hand and the PDF of
                    them, the transition table, the README (with what the
                    still notes, protocols\\grasps_still_notes.json, read
                    in the pictures by eye).
                    A file the build changes or no longer makes is moved to
                    `_superseded` beside the folder first, never deleted.
  3. THE CHECK      `leap_hand.grasp_validate`: the finished folder read
                    back and measured again; the report is printed and
                    written into the folder as validation_report.txt.

The exit code is 0 when the folder was written and its check passed. The
check's two statuses (the static-pose task, the multi-take collection) are
states of the work and do not change it. `--start-speed`, `--start-sustain`,
`--settle-speed`, `--settle-window` and `--cue-still` are the transition
timing's thresholds (`leap_hand.transition.transition_timing`); left out,
its defaults are used.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Sequence

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cam_hand import protocol_check as pc
from leap_hand import grasp_handin as gh
from leap_hand import grasp_validate as gv

HERE = Path(__file__).resolve().parent
THRESHOLDS = ("start-speed", "start-sustain", "settle-speed", "settle-window",
              "cue-still")


def script(name: str):
    """A sibling script as a module (`scripts/` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quoted(parts) -> str:
    return " ".join(f'"{a}"' if " " in str(a) else str(a) for a in parts)


def default_transitions_dir(session: Path) -> Path:
    """The `transitions` folder beside the session's own set folder:
    recordings/protocol/grasps/<session> -> recordings/protocol/transitions
    (and a mock session's recordings/protocol_mock/transitions)."""
    return Path(session).resolve().parent.parent / gh.TRANSITION_SET


def _same(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def discover_transition_sessions(folder: Path, allow_mock: bool = False,
                                 say: Callable[[str], None] = print
                                 ) -> List[Path]:
    """The transition sessions in `folder`, in name order: every immediate
    subfolder whose session.json is one of scripts/leap/record_transitions.py
    ("set": "transitions"). A mock one ("mock": true) only with
    `allow_mock`. Says one line per subfolder taken or passed over, and one
    ("no transition sessions in ...") when none is taken."""
    folder = Path(folder)
    subs = (sorted((p for p in folder.iterdir() if p.is_dir()),
                   key=lambda p: p.name) if folder.is_dir() else [])
    found: List[Path] = []
    for sub in subs:
        meta = pc.read_json(sub / "session.json")
        if not isinstance(meta, dict):
            say(f"  passed over {sub.name}: no readable session.json")
            continue
        if meta.get("set") != gh.TRANSITION_SET:
            say(f"  passed over {sub.name}: not a transition session "
                f"(set \"{meta.get('set')}\")")
            continue
        mock = bool(pc.mock_flags(meta))
        if mock and not allow_mock:
            say(f"  skipped {sub.name}: a mock transition session (synthetic "
                "hand, recorded with --mock); taken only with --allow-mock")
            continue
        kept = sum(1 for t in meta.get("takes") or []
                   if isinstance(t, dict) and t.get("accepted"))
        say(f"  transition session {sub.name}: {kept} kept take(s)"
            + (", MOCK (synthetic hand)" if mock else ""))
        found.append(sub)
    if not found:
        say(f"  no transition sessions in {folder}"
            + (" (no such folder)" if not folder.is_dir()
               else " to add" if subs else ""))
    return found


def transition_folders(given: Sequence[Path], folder: Path,
                       allow_mock: bool = False,
                       say: Callable[[str], None] = print) -> List[Path]:
    """The packager's --transitions: the folders given, as given, then the
    sessions found in `folder` (`discover_transition_sessions`), none twice."""
    out = list(given)
    seen = {_same(p) for p in out}
    for sub in discover_transition_sessions(folder, allow_mock, say):
        if _same(sub) in seen:
            say(f"  ({sub.name} is also given with --transitions: added once)")
            continue
        seen.add(_same(sub))
        out.append(sub)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Rebuild the professor's grasp folder from a Set A "
                    "session: joint frames, the folder, the check.")
    p.add_argument("--session", type=Path, required=True,
                   help="the Set A session folder (session.json)")
    p.add_argument("--out", type=Path, required=True,
                   help="the hand-in folder to write")
    p.add_argument("--transitions", type=Path, nargs="+", action="extend",
                   default=[],
                   help="transition session folder(s) of scripts/leap/"
                        "record_transitions.py to add to the transition "
                        "table")
    p.add_argument("--transitions-dir", type=Path, default=None,
                   help="a folder whose transition sessions are all added "
                        "too, mock ones only with --allow-mock (default: "
                        "the transitions folder beside the session's set "
                        "folder, recordings/protocol/transitions for a "
                        "session in recordings/protocol/grasps)")
    p.add_argument("--operator", default=None,
                   help="operator name for the README (default: from "
                        "session.json)")
    p.add_argument("--participant", default="P01",
                   help="participant id in the transition table")
    p.add_argument("--frame-ref", default="wrist", choices=("wrist", "palm"),
                   help="reference frame of the joint-frame tables "
                        "(default: the wrist joint)")
    for name in THRESHOLDS:
        p.add_argument(f"--{name}", type=float, default=None,
                       help="transition timing threshold (see the docstring)")
    p.add_argument("--allow-mock", action="store_true",
                   help="build from a mock (synthetic) session too; only for "
                        "rehearsing, never for the hand-in")
    args = p.parse_args(argv)
    raw = list(argv) if argv is not None else sys.argv[1:]
    if not (args.session / "session.json").is_file():
        print(f"no session.json in {args.session}")
        return 2

    print(f"1/3 joint frames of {args.session.name}")
    view = script("joint_frames_view")
    try:
        result = view.export_session(args.session, args.frame_ref)
    except FileNotFoundError as e:
        print(f"not a session folder: {e}")
        return 2
    for text in result["skipped"]:
        print(f"  skipped {text}")
    if result["pdf"] is None:
        print("  no accepted take could be exported; nothing built")
        return 1
    print(f"  {len(result['pages'])} table(s) and {result['pdf'].name}")

    print(f"2/3 the folder {args.out}")
    transitions = transition_folders(
        args.transitions,
        args.transitions_dir or default_transitions_dir(args.session),
        args.allow_mock)
    superseded = (args.out.parent / "_superseded" / args.out.name
                  / time.strftime("%Y%m%d_%H%M%S"))
    package = script("package_professor_set")
    command = quoted(["python", "scripts/rebuild_grasp_set.py"] + raw)
    call = ["--out", str(args.out), "--grasps", str(args.session),
            "--participant", args.participant, "--command", command,
            "--superseded", str(superseded)]
    if args.operator:
        call += ["--operator", args.operator]
    if transitions:
        call += ["--transitions"] + [str(t) for t in transitions]
    if args.allow_mock:
        call.append("--allow-mock")
    for name in THRESHOLDS:
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            call += [f"--{name}", str(value)]
    rc = package.main(call)
    if rc != 0:
        print("the folder was not built; nothing checked")
        return rc

    print("3/3 the check")
    # --allow-mock reaches the check too: without it a mock transition
    # session's rows (MOCK in their notes) fail the folder
    report = gv.validate(args.out, args.session, allow_mock=args.allow_mock)
    text = gv.render(report)
    dest = gh.long_path(args.out / gh.VALIDATION_FILE)
    old = dest.read_text(encoding="utf-8") if dest.is_file() else None
    if old != text:
        if old is not None:
            package.set_aside(dest, args.out, superseded)
        dest.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
