"""Quality numbers for Leap recordings: the Phase 2 gate, as a table.

One row per hand per file: how much of the hand the camera actually saw, how
often it lost and re-found it, how fast it was tracking, and how much the
fingertips wobble when the hand is still. Those four are what decides Path A
(simultaneous glove + camera capture) against Path B (sequential), so this is
the script the gate experiment is written around.

Column by column (definitions live in `leap_hand.stats`):

  n        frames of that hand in the file
  span     seconds from its first frame to its last
  rate     the denominator det% is measured against, marked * when it is the
           LeapC tracking framerate (the recorder kept every frame, so every
           gap IS one tracking interval) and unmarked when it is the file's
           own cadence (a --hz throttled file). `leap_hand.stats.choose_rate`
           decides, and its docstring says why the choice matters
  cad      that cadence: the 10th-percentile gap between frames of that hand
  fps      mean tracking framerate LeapC reported, never assumed to be 90
  det%     frames present / (span x rate): unrecovered dropouts
  reacq    hand-id changes: losses the tracker recovered from
  jit_mm   fingertip spread over the steadiest 2 s, in millimetres
  age_ms   frame age at receipt (leap.get_now() - event.timestamp), not
           end-to-end latency; blank for mock recordings, which have no
           LeapC clock behind them

Usage:
  python scripts/leap/stats.py recordings/leap/poses
  python scripts/leap/stats.py recordings/leap/poses --write
  python scripts/leap/stats.py a.jsonl b.jsonl --sort det
"""
import argparse
from pathlib import Path

from leap_hand.stats import analyse_paths, format_table

HERE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("recordings") / "leap" / "poses"

SORT_KEYS = {
    "file": lambda s: (s.file, s.hand_side),
    "det": lambda s: s.detection_rate,
    "reacq": lambda s: -s.reacquisitions,
    "jitter": lambda s: -s.jitter_mm,
    "fps": lambda s: s.mean_framerate,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Detection rate, re-acquisitions, framerate and jitter "
                    "for Leap recordings.")
    p.add_argument("input", type=Path, nargs="*", default=[DEFAULT_INPUT],
                   help=f".jsonl files or folders (default: {DEFAULT_INPUT})")
    p.add_argument("--write", action="store_true",
                   help="also write results/leap_stats.txt")
    p.add_argument("--out", type=Path, default=None,
                   help="where --write puts the table "
                        "(default: results/leap_stats.txt)")
    p.add_argument("--sort", choices=sorted(SORT_KEYS), default="file",
                   help="row order (default: file)")
    p.add_argument("--mock", action="store_true",
                   help="analyse a freshly generated mock recording instead "
                        "of reading files (a self-test of this script)")
    args = p.parse_args()

    if args.mock:
        rows = mock_rows()
        title = "mock recording (10 s of synthetic hands)"
    else:
        missing = [str(p) for p in args.input if not p.exists()]
        if missing:
            raise SystemExit("no such path: " + ", ".join(missing))
        rows = analyse_paths(args.input)
        title = ", ".join(str(p) for p in args.input)
        if not rows:
            raise SystemExit(f"no .jsonl recordings in {title}")

    rows.sort(key=SORT_KEYS[args.sort])
    table = f"Leap recording stats - {title}\n\n" + format_table(rows)
    print()
    print(table)

    if args.write:
        out = args.out or (HERE / "results" / "leap_stats.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


def mock_rows():
    """Record 10 s of mock hands to a temp file and analyse that."""
    import tempfile

    from leap_hand.mock import MockLeapStream
    from leap_hand.recorder import LeapRecorder

    stream = MockLeapStream()
    stream.start()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mock_both_take1.jsonl"
        rec = LeapRecorder(pose="mock", take=1)
        rec.start(path)
        for _side, lh in stream.generate(900):      # 10 s at 90 Hz
            rec.record(lh)
        rec.stop()
        return analyse_paths([path])


if __name__ == "__main__":
    main()
