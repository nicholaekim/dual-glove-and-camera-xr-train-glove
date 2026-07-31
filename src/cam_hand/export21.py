"""Shared conversion helpers for camera recordings.

Everything downstream (CSV export, professor-format export, pose analysis)
uses the same convention as the glove pipeline: wrist-centred world XYZ in
metres, 21 keypoints in MediaPipe order.
"""
import time
from pathlib import Path
from typing import Generator, List

from .recorder import CamRecorder

# (index, finger label as in the professor's tracker files). Landmark 0 in his
# layout is the palm — MediaPipe has no palm point, so it is synthesized as the
# wrist / middle-knuckle midpoint. Indices 1..20 correspond 1:1 to MediaPipe.
PROF_LABELS = (
    ["Palm"] + ["THUMB"] * 4 + ["Index"] * 4 + ["Middle"] * 4
    + ["Ring"] * 4 + ["Pinky"] * 4
)

MM = 1000.0  # world landmarks are metres; the tracker files use mm


def iter_cam_files(input_path: Path) -> List[Path]:
    """A .jsonl file, or every camera .jsonl under a folder."""
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.jsonl"))
    return [input_path]


def iter_records(paths) -> Generator[dict, None, None]:
    for path in paths:
        for d in CamRecorder.load(path):
            if "world" in d:   # skip non-camera jsonl that may share the folder
                yield d


def wrist_centered(d: dict) -> List[List[float]]:
    """Record dict -> 21 [x, y, z] world points in metres, wrist at origin."""
    w = d["world"]
    wx, wy, wz = w[0]
    return [[x - wx, y - wy, z - wz] for x, y, z in w]


def palm_point(pts: List[List[float]]) -> List[float]:
    """Palm approximated as the wrist / middle-MCP midpoint (landmarks 0, 9)."""
    return [(pts[0][i] + pts[9][i]) / 2.0 for i in range(3)]


def prof_block(d: dict, frame_no: int) -> str:
    """One camera record -> a professor-format text block (mm, wrist origin)."""
    pts = wrist_centered(d)
    stamp = (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(d["wall_time"]))
             + f".{int(d['wall_time'] * 1000) % 1000:03d}")
    lines = [f"Frame {frame_no} | Hand ID: {d['hand_side']} | Time: {stamp}"]
    lines.append("Wrist: (0.00, 0.00, 0.00)")
    coords = [palm_point(pts)] + pts[1:]
    for idx, (label, p) in enumerate(zip(PROF_LABELS, coords)):
        lines.append(f"{idx} ({label}): ({p[0] * MM:.2f}, {p[1] * MM:.2f}, {p[2] * MM:.2f})")
    return "\n".join(lines)
