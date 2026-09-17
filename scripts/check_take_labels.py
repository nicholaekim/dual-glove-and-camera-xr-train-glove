"""Does every take in a session folder actually hold the pose it is named after?

The audit half of `leap_hand.pose_check`. The recorder now refuses a take
whose hand is not the requested pose, but the sessions already on disk were
recorded before it did, and one of them (`recordings/sync_coached_20260917`,
36 takes) has nine takes holding a pose that is not the one in the filename —
with BOTH sensors agreeing on the wrong one, because the live camera window
never named the pose the operator was supposed to be making.

Run it over a folder and it prints, per take: the verdict, both sensors'
median finger curls, how healthy the glove stream was, and one line of
English saying what the sensors actually saw. Then, per hand, the exact
command to re-record the poses that were wrong.

  python scripts/check_take_labels.py recordings/sync_coached_20260917
  python scripts/check_take_labels.py recordings/sync --camera leap

Takes under `<folder>/rejected/` are NOT audited: they are attempts the
recorder already threw out, kept on disk so the exclusions can be counted,
and auditing them would count every one of them twice.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leap_hand.pose_check import (
    DEFAULT_PARAMS,
    MISMATCH,
    OK,
    UNCHECKED,
    WARN,
    PoseCheckParams,
    check_pose,
    read_take,
)
from leap_hand.protocol import stream_health

CAM_DIRS = ("leap", "cam")
AUTO = "auto"
ORDER = {MISMATCH: 0, WARN: 1, UNCHECKED: 2, OK: 3}


def take_label(path: Path):
    """(pose, take, hand) for one recording, read out of the file itself.

    Off the first line, not off the filename: a file that was renamed or
    moved still says what it is, and `pose`/`take`/`hand_side` are written
    into every single frame by both recorders.
    """
    pose, take, hand = "", None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    pose = d.get("pose", "") or ""
                    take = d.get("take")
                    hand = d.get("hand_side")
                    break
    except (OSError, ValueError):
        pass
    if not pose:                                  # fall back to the name
        stem = path.name.split("_take")[0]
        for tag in ("_left", "_right", "_both"):
            if stem.endswith(tag):
                stem, hand = stem[:-len(tag)], tag[1:]
        pose = stem
    return pose, take, hand


def camera_file(folder: Path, name: str, camera: str):
    for sub in ((camera,) if camera != AUTO else CAM_DIRS):
        candidate = folder / sub / name
        if candidate.is_file():
            return candidate
    return None


def curl_text(curls) -> str:
    if not curls:
        return f"{'-':^23}"
    return " ".join(f"{v:4.2f}" for v in curls)


def health_text(health) -> str:
    if not health or health.get("rate_hz") is None:
        return f"{health['frames']:>3}f" if health else ""
    over = health["gaps_over"]
    return (f"{health['frames']:>3}f {health['rate_hz']:5.1f} Hz  "
            f"worst gap {health['max_gap_ms']:6.1f} ms"
            + (f"  ({over} over {health['gaps_over_ms']:.0f} ms)" if over else ""))


def audit(folder: Path, camera: str, params: PoseCheckParams):
    """Every take in `folder`, checked. Returns a list of result dicts."""
    glove_dir = folder / "glove"
    if not glove_dir.is_dir():
        raise SystemExit(f"no glove takes in {glove_dir}\n"
                         "  point this at a session folder holding glove/ "
                         "plus leap/ or cam/")
    out = []
    for gpath in sorted(glove_dir.glob("*.jsonl")):
        cpath = camera_file(folder, gpath.name, camera)
        pose, take, hand = take_label(gpath)
        glove = read_take(gpath, hand)
        cam = read_take(cpath, hand) if cpath else read_take(None)
        check = check_pose(pose, glove, cam, params)
        span = (max(cam.times) - min(cam.times)) if len(cam.times) > 1 else None
        out.append({
            "name": gpath.name, "stem": gpath.stem, "pose": pose, "take": take,
            "hand": hand or "?", "check": check,
            "glove_health": stream_health(glove.times),
            "camera_health": stream_health(cam.times),
            "span_s": span,
            "camera_file": cpath,
        })
    return out


def redo_commands(rows) -> list:
    """Per hand, the one command that re-records every pose that was wrong."""
    bad = defaultdict(set)
    takes_of = defaultdict(set)
    spans = []
    for r in rows:
        takes_of[(r["hand"], r["pose"])].add(r["take"])
        if r["span_s"]:
            spans.append(r["span_s"])
        if r["check"].verdict == MISMATCH:
            bad[r["hand"]].add(r["pose"])
    duration = round(sorted(spans)[len(spans) // 2]) if spans else 5
    lines = []
    for hand in sorted(bad):
        poses = sorted(bad[hand])
        takes = max((len(takes_of[(hand, p)]) for p in poses), default=1)
        lines.append(
            f"  {hand:<6} python scripts\\record_simultaneous.py --camera leap "
            f"--hand {hand} --poses {','.join(poses)} "
            f"--takes {takes} --duration {duration:g}")
    return lines


def report(rows, params: PoseCheckParams, folder: Path) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"Pose check — {len(rows)} take(s) in {folder}")
    lines.append("")
    lines.append("  curls are tip-to-wrist over palm length (thumb, index, "
                 "middle, ring, pinky),")
    lines.append("  median over the take — the same number the per-DOF table "
                 "in fuse_poses.py prints.")
    lines.append("  A finger is only WRONG when both sensors are decisive and "
                 "both contradict the")
    lines.append("  pose; one sensor alone is a warn. pinch is measured on the "
                 "camera's thumb-index")
    lines.append("  gap and is never rejected on it.")
    lines.append("")
    lines.append(f"  {'take':<34} {'verdict':<9} {'glove curls':<24} "
                 f"{'camera curls':<24}")
    lines.append(f"  {'-' * 34} {'-' * 9} {'-' * 24} {'-' * 24}")
    counts = defaultdict(int)
    for r in rows:
        c = r["check"]
        counts[c.verdict] += 1
        lines.append(f"  {r['stem'][:34]:<34} {c.verdict.upper():<9} "
                     f"{curl_text(c.glove_curls):<24} "
                     f"{curl_text(c.camera_curls):<24}")
        lines.append(f"      {c.description}")
        lines.append(f"      glove  {health_text(r['glove_health'])}"
                     f"   |  camera {health_text(r['camera_health'])}")
    lines.append("")
    lines.append("  " + "   ".join(
        f"{v}: {counts.get(v, 0)}" for v in (OK, WARN, MISMATCH, UNCHECKED)))

    redo = redo_commands(rows)
    lines.append("")
    if redo:
        lines.append("Re-record these WHOLE POSES. Take numbering is per pose, "
                     "so a pose is re-recorded")
        lines.append("from take 1 — you cannot re-record take 2 of fist on its "
                     "own, and the old takes")
        lines.append("of that pose should be moved aside first.")
        lines.extend(redo)
    else:
        lines.append("No take was contradicted by both sensors: nothing to "
                     "re-record.")
    lines.append("")
    lines.append("Thresholds in force (leap_hand.pose_check.PoseCheckParams)")
    for k, v in params.described().items():
        lines.append(f"  {k:<22} {v}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check every take in a session folder against the pose it "
                    "is labelled with.")
    p.add_argument("input", type=Path, nargs="?",
                   default=Path("recordings") / "sync",
                   help="session folder holding glove/ plus leap/ or cam/")
    p.add_argument("--camera", choices=(AUTO,) + CAM_DIRS, default=AUTO,
                   help="which camera folder to read (default: auto)")
    p.add_argument("--write", action="store_true",
                   help="also write <input>/POSE_CHECK.txt")
    p.add_argument("--json", type=Path, default=None,
                   help="also write the verdicts as JSON")
    args = p.parse_args()

    rows = audit(args.input, args.camera, DEFAULT_PARAMS)
    if not rows:
        raise SystemExit(f"no takes found in {args.input / 'glove'}")
    text = report(rows, DEFAULT_PARAMS, args.input)
    print(text)
    if args.write:
        out = args.input / "POSE_CHECK.txt"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            [{"take": r["stem"], "pose": r["pose"], "hand": r["hand"],
              "glove_stream": r["glove_health"],
              "camera_stream": r["camera_health"],
              **r["check"].as_dict()} for r in rows], indent=2) + "\n",
            encoding="utf-8")
        print(f"wrote {args.json}")
    raise SystemExit(1 if any(r["check"].verdict == MISMATCH for r in rows)
                     else 0)


if __name__ == "__main__":
    main()
