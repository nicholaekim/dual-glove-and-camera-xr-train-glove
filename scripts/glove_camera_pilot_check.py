"""Inspect XR Train OSC addresses or audit paired glove/Ultraleap holds.

Reuses the existing joint decomposition and pilot_compare.measure. The audit
reports disagreement between solved skeletons, with no anatomical accuracy
claim and no inference about how many physical sensors the gloves contain.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pythonosc.osc_packet import OscPacket
from cam_hand.recorder import is_settle_file
from xr_hand import joint_frames as jf
import pilot_compare


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_hand(path, hand):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()
            if s.strip()]
    selected = [r for r in rows if r.get("hand_side") == hand]
    if not selected:
        raise ValueError(f"{path}: no {hand}-hand frames")
    if any(r.get("status") == 0 for r in selected):
        raise ValueError(f"{path}: invalid tracking present; this hold needs review")
    return selected


def angle_ranges(rows, hand):
    values = {}
    for line in rows:
        if line.get("hand_side") != hand:
            raise ValueError("angle range input contains another hand")
        for row in jf.joint_table(line, "glove", reference="palm", hand=hand):
            for component in ("flex_deg", "abd_deg", "twist_deg"):
                v = float(row[component])
                if not math.isfinite(v):
                    raise ValueError("non-finite joint angle")
                key = row["joint"], component
                bounds = values.setdefault(key, [v, v])
                bounds[0], bounds[1] = min(bounds[0], v), max(bounds[1], v)
    return [{"joint": j, "component": c, "min_deg": a, "max_deg": b,
             "span_deg": b - a, "frames": len(rows)}
            for (j, c), (a, b) in values.items()]


def write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with Path(path).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)


def new_output(path, inputs=()):
    path = Path(path).resolve()
    for source in inputs:
        if path == source.resolve() or source.resolve() in path.parents:
            raise ValueError("output must be outside the source recording folder")
    path.mkdir(parents=True, exist_ok=False)
    return path


def audit(session, out, hand="left", poses=("open_palm", "fist", "pinch"),
          origin="recorded", window=3.0):
    session = Path(session).resolve()
    if not (session / "glove").is_dir() or not (session / "leap").is_dir():
        raise ValueError("session needs glove/ and leap/ folders from record_simultaneous.py")
    if not math.isfinite(window) or window <= 0:
        raise ValueError("window must be positive seconds")
    takes, sources, all_glove = [], [], []
    for pose in poses:
        files = sorted(p for p in (session / "glove").glob(f"{pose}_{hand}_take1_*.jsonl")
                       if not is_settle_file(p))
        if len(files) != 1:
            raise ValueError(f"{pose}: expected exactly one {hand} take 1, found {len(files)}")
        glove_path = files[0]
        cam_path = session / "leap" / glove_path.name
        glove, cam = load_hand(glove_path, hand), load_hand(cam_path, hand)
        for rows in (glove, cam):
            if any(r.get("pose") != pose or r.get("take") != 1 for r in rows):
                raise ValueError(f"{pose}: source labels disagree with the selected take")
        all_glove.extend(glove)
        meta_path = cam_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        row = {"pose": pose, "file": glove_path.name, "hand": hand,
               "origin": origin, "glove_frames": len(glove), "camera_frames": len(cam)}
        row["status"] = pilot_compare.measure(row, session, meta.get("hold_end"), window, 0.05)
        takes.append(row)
        for path in (glove_path, cam_path, *([meta_path] if meta_path.exists() else [])):
            sources.append({"path": str(path), "sha256": sha256(path)})
    out = new_output(out, (session,))
    ranges = angle_ranges(all_glove, hand)
    report = {"source_session": str(session), "hand": hand, "data_origin": origin,
              "physical_raw_sensor_count": None, "physical_sensor_mapping": "unconfirmed",
              "interpretation": "Solved-skeleton disagreement; neither device is ground truth.",
              "timing": "capture_time is camera acquisition versus glove OSC receipt. "
                        "wall_time is the fallback. Neither establishes device latency or "
                        "physical synchronization uncertainty. No lag correction fitted.",
              "window_s": window, "max_pair_dt_s": 0.05, "sources": sources,
              "takes": takes, "glove_angle_ranges": ranges}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out / "takes.csv", takes)
    write_csv(out / "glove_angle_ranges.csv", ranges)
    print(f"PILOT AUDIT OK: pairs={len(takes)} origin={origin} out={out}")
    for row in takes:
        print(f"  {row['pose']}: {row['status']}; clock={row.get('clock')}; "
              f"pairs={row.get('n_pairs', 0)}; median disagreement across 8 joints="
              f"{row.get('absdiff_med_all8')}")
    return report


def probe(seconds, out):
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("seconds must be greater than 0 and at most 60")
    counts, labels, malformed = {}, set(), 0
    start = time.perf_counter()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 9002))
        sock.settimeout(min(seconds, 0.2))
        while time.perf_counter() - start < seconds:
            try:
                data = sock.recv(65535)
            except socket.timeout:
                continue
            try:
                packet = OscPacket(data)
                for timed in packet.messages:
                    msg = timed.message
                    entry = counts.setdefault(msg.address, {"messages": 0, "argument_counts": []})
                    entry["messages"] += 1
                    if len(msg.params) not in entry["argument_counts"]:
                        entry["argument_counts"].append(len(msg.params))
                    if msg.address == "/v1/animation/kinematic/all" and len(msg.params) >= 5:
                        labels.update(x for x in msg.params[3:5] if isinstance(x, str))
            except Exception:
                malformed += 1
    report = {"seconds": time.perf_counter() - start, "endpoint": "127.0.0.1:9002",
              "addresses": counts, "device_labels": sorted(labels), "malformed_packets": malformed,
              "physical_sensor_mapping": "Not inferred from OSC argument counts."}
    out = new_output(out)
    (out / "osc_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"OSC PROBE: addresses={len(counts)} labels={sorted(labels)} out={out}")
    return bool(counts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--session", type=Path, required=True)
    a.add_argument("--hand", choices=("left", "right"), default="left")
    a.add_argument("--poses", default="open_palm,fist,pinch")
    a.add_argument("--data-origin", choices=("recorded", "synthetic"), default="recorded")
    a.add_argument("--window", type=float, default=3.0)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("probe")
    b.add_argument("--seconds", type=float, default=5.0)
    b.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "probe":
            return 0 if probe(args.seconds, args.out) else 2
        poses = tuple(x.strip() for x in args.poses.split(",") if x.strip())
        if not poses or any(not x.replace("_", "").isalnum() for x in poses):
            raise ValueError("poses must be comma-separated pose identifiers")
        report = audit(args.session, args.out, args.hand, poses, args.data_origin, args.window)
        return 0 if all(t["status"] == "ok" for t in report["takes"]) else 2
    except (ValueError, OSError) as e:
        print(f"PILOT CHECK FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
