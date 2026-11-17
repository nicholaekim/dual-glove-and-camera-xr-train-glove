r"""The 100-pose catalogue as a recorder protocol, with a picture for every pose.

Reads protocols/pose_catalogue_100.csv and results/pose_catalogue/renders/
(written by pose_catalogue.py and pose_catalogue_render.py) and writes, for
each hand:

    results/pose_catalogue/panels_<hand>/<id>.png   palm view and side view side by side
    protocols/pose_catalogue_100_<hand>.json        items with id, label, image

so that scripts/leap/record_poses.py --protocol shows each pose's picture in
its COPY THIS window. The renders are of a right hand; the left-hand panels
are the same pictures mirrored. They are MyoHand model renders, not
photographs. A third set, panels_both/ and pose_catalogue_100_both.json, is
for record_poses.py --both: both palms side by side, the side view under them.

    .venv\Scripts\python.exe scripts\leap\pose_catalogue_protocol.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "protocols" / "pose_catalogue_100.csv"
RENDERS = REPO / "results" / "pose_catalogue" / "renders"


TEXT_H = 380                  # the band under the two views, for the description
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _wrap(text: str, scale: float, thick: int, width: int) -> list:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and cv2.getTextSize(trial, FONT, scale, thick)[0][0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def _view(pid: str, v: str, left: bool):
    img = cv2.imread(str(RENDERS / f"{pid}_{v}.png"))
    if img is None:
        raise SystemExit(f"no render for {pid} ({v}): run pose_catalogue_render.py first")
    # the renders have the fingers pointing sideways; turn them fingers-up
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if v == "palm" else cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.flip(img, 1) if left else img       # the renders are of a right hand


def panel_both(pid: str, description: str):
    """For a --both take: the left and the right palm side by side, and under them the
    side view with the description beside it."""
    top = np.hstack([_view(pid, "palm", True), _view(pid, "palm", False)])
    side = _view(pid, "side", False)
    h, w = side.shape[:2]
    for img, caption in ((top[:, :w], "left hand"), (top[:, w:], "right hand"), (side, "side view")):
        cv2.putText(img, caption, (14, 30), FONT, 0.8, (120, 120, 120), 2, cv2.LINE_AA)
    text = np.full((h, w, 3), 255, np.uint8)
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6):
        lines = _wrap(description, scale, 2, w - 30)
        if len(lines) * int(46 * scale) <= h - 60:
            break
    for k, line in enumerate(lines):
        cv2.putText(text, line, (12, 60 + k * int(46 * scale)), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([top, np.hstack([side, text])])


def panel(pid: str, hand: str, description: str):
    """Palm view and side view side by side, the pose's description in words under them:
    some poses differ by less than the picture can show (a few degrees of spread)."""
    if hand == "both":
        return panel_both(pid, description)
    top = np.hstack([_view(pid, v, hand == "left") for v in ("palm", "side")])
    w = top.shape[1]
    band = np.full((TEXT_H, w, 3), 255, np.uint8)
    for k, caption in enumerate(("palm view", "side view")):
        cv2.putText(band, caption, (k * w // 2 + 16, 30), FONT, 0.8, (120, 120, 120), 2, cv2.LINE_AA)
    for scale in (1.3, 1.15, 1.0, 0.9, 0.8):
        lines = _wrap(description, scale, 2, w - 40)
        if len(lines) * int(46 * scale) <= TEXT_H - 80:
            break
    for k, line in enumerate(lines):
        cv2.putText(band, line, (20, 90 + k * int(46 * scale)), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([top, band])


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for hand in ("left", "right", "both"):
        out = REPO / "results" / "pose_catalogue" / f"panels_{hand}"
        out.mkdir(parents=True, exist_ok=True)
        items = []
        for r in rows:
            cv2.imwrite(str(out / f"{r['id']}.png"), panel(r["id"], hand, r["description"]))
            items.append({"id": r["id"], "label": f"{r['id']} {r['name']}", "group": r["category"],
                          "shape": r["description"], "orientation": r["description"],
                          "image": f"{r['id']}.png"})
        proto = {
            "name": f"pose_catalogue_100_{hand}",
            "version": 1,
            "description": "The 100 catalogue poses of docs/pose_catalogue_100.md, camera only, bare "
                           + ("hands, both making the pose in one take (record_poses.py --both)"
                              if hand == "both" else f"{hand} hand")
                           + "; pictures are MyoHand renders (palm view, side view), not "
                           "photographs. Written by scripts/leap/pose_catalogue_protocol.py.",
            "setup": rows[0]["setup"],
            "images_dir": f"../results/pose_catalogue/panels_{hand}",
            "takes_per_item": 1, "duration_s": 5.0, "prep_s": 5.0,
            "items": items,
        }
        path = REPO / "protocols" / f"pose_catalogue_100_{hand}.json"
        path.write_text(json.dumps(proto, indent=1), encoding="utf-8")
        print(f"PROTOCOL OK: {path.name} items={len(items)} panels={out}")
        if hand == "both":
            continue
        # The 50/50 split: poses alternate between the hands in catalogue order (1st, 3rd, ...
        # left; 2nd, 4th, ... right), so each hand gets poses from every category.
        half = dict(proto, name=f"pose_catalogue_50_{hand}", items=items[(0 if hand == "left" else 1)::2])
        half["description"] = (f"Half of the 100 catalogue poses, the ones assigned to the {hand} hand "
                               "(alternating in catalogue order); the other half is in the other "
                               "hand's file. " + proto["description"])
        path = REPO / "protocols" / f"pose_catalogue_50_{hand}.json"
        path.write_text(json.dumps(half, indent=1), encoding="utf-8")
        print(f"PROTOCOL OK: {path.name} items={len(half['items'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
