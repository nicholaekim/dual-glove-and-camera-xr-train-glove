"""Read the professor's tracker keypoint text files.

His format (one block per frame, 21 landmarks, millimetres in the tracker's
own camera space):

    Frame 102287 | Hand ID 1232 (right)
    0 (Palm): (-126.94, 247.02, 1.67)
    1 (THUMB): (-163.28, 267.95, 39.59)
    ... through ...
    20 (Pinky): (-109.00, 260.55, 20.88)

Landmark 0 is the palm; each finger then contributes four points from the
knuckle outward. Files written by this project's exporters use a slightly
richer header (`Hand ID: right | Time: ...`) plus a `Wrist:` line — both are
accepted here, so the same reader handles his data and ours.
"""
import re
from pathlib import Path
from typing import Iterator, List, NamedTuple

# "Frame 102287 | Hand ID 1232 (right)"  and  "Frame 3 | Hand ID: right | Time: ..."
HEADER_RE = re.compile(
    r"^Frame\s+(?P<frame>\d+)\s*\|\s*Hand ID:?\s*(?P<id>\d+)?\s*"
    r"\(?(?P<hand>left|right)\)?", re.IGNORECASE)
POINT_RE = re.compile(
    r"^(?P<idx>\d+)\s*\((?P<label>[^)]+)\):\s*\(\s*(?P<x>-?\d+\.?\d*)\s*,"
    r"\s*(?P<y>-?\d+\.?\d*)\s*,\s*(?P<z>-?\d+\.?\d*)\s*\)")

N_LANDMARKS = 21


class ProfFrame(NamedTuple):
    frame: int
    hand: str
    hand_id: str
    points: List[List[float]]   # 21 x [x, y, z] in millimetres, index 0 = palm
    source: str


def parse_text(text: str, source: str = "") -> Iterator[ProfFrame]:
    """Yield every frame block found in the text."""
    frame = hand = hand_id = None
    points: dict = {}

    def flush():
        if frame is not None and len(points) == N_LANDMARKS:
            return ProfFrame(frame, hand, hand_id,
                             [points[i] for i in range(N_LANDMARKS)], source)
        return None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = HEADER_RE.match(line)
        if m:
            done = flush()
            if done is not None:
                yield done
            frame = int(m.group("frame"))
            hand = m.group("hand").lower()
            hand_id = m.group("id") or ""
            points = {}
            continue
        m = POINT_RE.match(line)
        if m:
            idx = int(m.group("idx"))
            if 0 <= idx < N_LANDMARKS:
                points[idx] = [float(m.group("x")), float(m.group("y")),
                               float(m.group("z"))]
    done = flush()
    if done is not None:
        yield done


def load_file(path: str | Path) -> List[ProfFrame]:
    path = Path(path)
    return list(parse_text(path.read_text(encoding="utf-8", errors="replace"),
                           source=path.name))


def find_pairs(root: str | Path) -> List[tuple]:
    """Find (image, keypoint-txt) pairs in his dataset layout.

    Handles both `frame_<n>_DONE/frame_<n>.png` + `.txt` folders and a flat
    folder of matching `.png` / `.txt` stems.
    """
    root = Path(root)
    pairs = []
    for txt in sorted(root.rglob("*.txt")):
        for ext in (".png", ".jpg", ".jpeg"):
            img = txt.with_suffix(ext)
            if img.is_file():
                pairs.append((img, txt))
                break
    return pairs
