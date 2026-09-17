"""Export a JSONL recording to CSV for analysis / sharing.

Each row is one frame. Columns:
  wall_time         Unix epoch seconds when the frame was received
  iso_time          human-readable local time, ms precision
  timestamp         XR Trainer's internal tick counter
  packet_counter, hand_side, frame_id, status
  pose, take        gesture label + repetition (empty for unlabeled sessions)
  <JOINT>_x, _y, _z, _qw, _qx, _qy, _qz  (26 joints x 7 = 182 columns)

Positions and quaternions are kept in the device's native local frame —
forward kinematics is left to the analyst (joints.BONES + kinematics.py
do this in 5 lines if needed).

    python scripts/export.py recordings/my_session.jsonl
    python scripts/export.py recordings/my_session.jsonl --hand left
    python scripts/export.py recordings/my_session.jsonl --output out.csv
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from xr_hand.joints import JOINT_NAMES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="Path to .jsonl recording")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .csv path (defaults to same name as input)")
    p.add_argument("--hand", choices=["left", "right"], default=None,
                   help="Filter to one hand only (default: include both)")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"file not found: {args.input}")

    output = args.output or args.input.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "wall_time", "iso_time", "timestamp", "packet_counter",
        "hand_side", "frame_id", "status", "pose", "take",
    ]
    for name in JOINT_NAMES:
        for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz"):
            header.append(f"{name}_{suffix}")

    written = 0
    skipped = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if args.hand and d["hand_side"] != args.hand:
                skipped += 1
                continue
            iso = datetime.fromtimestamp(d["wall_time"]).isoformat(
                sep=" ", timespec="milliseconds"
            )
            row = [
                d["wall_time"], iso, d["timestamp"], d["packet_counter"],
                d["hand_side"], d["frame_id"], d["status"],
                d.get("pose", ""), d.get("take", ""),
            ]
            for j in d["joints"]:
                row.extend([j["x"], j["y"], j["z"],
                            j["qw"], j["qx"], j["qy"], j["qz"]])
            writer.writerow(row)
            written += 1

    print(f"wrote {written} rows to {output}")
    if skipped:
        print(f"skipped {skipped} rows (hand filter)")


if __name__ == "__main__":
    main()
