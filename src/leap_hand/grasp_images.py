"""Hand-only pictures for a hand-in folder, cut from the recorder's stills.

The recorder keeps one infrared picture per take (`stills/<take>.png`,
`scripts/leap/record_poses.py`): the camera's picture at the midpoint of the
hold, already cropped to the tracked hands by the viewer, mirrored as the
operator saw it, with the tracker's skeleton drawn on it (the viewer's
colours: a hand labelled left in cyan, one labelled right in red). The
camera looks up past the hand at whoever is holding it, so most stills have
the operator's face in them, and a still is never handed in as it is.

`hand_only` keeps the recorded hand and blacks out the rest:

  the skeleton   the pixels drawn in the viewer's colour for the recorded
                 hand's label: inside `SKELETON_BAND[hand]`, inclusive RGB
                 ranges around the colour, because the lines are
                 anti-aliased (`skeleton_mask`);
  the hand's     the median distance of those pixels from their median
  size           point (`radius`): the fingers' many lines outweigh the one
                 line drawn along the forearm;
  kept           every pixel within `CORE` radii of a skeleton pixel, and
                 every pixel within `REACH` radii that is at least `GATE`
                 times as bright as the hand itself (the median brightness,
                 mean of R, G and B, of the undrawn pixels within
                 `LEVEL_PX` of the skeleton; undrawn = in neither the
                 skeleton's band nor the caption's `CAPTION_BAND`). The
                 distance is OpenCV's distance transform (L2, a
                 `DISTANCE_MASK` x `DISTANCE_MASK` mask). The hand is nearer
                 the camera's own lights than anything behind it, so it is
                 the brightest thing in the picture;
  closed         the kept set closed with a `CLOSE_PX` x `CLOSE_PX` window
                 (dilated, then eroded): a hole or gap narrower than that
                 between kept pixels is filled, so a few pixels the two
                 tests alone would drop stay;
  one piece      of that, only what is connected to the skeleton (8
                 neighbours) stays.

Everything else is set to black and the picture is cut to what stayed. What
is left is the hand, a strip of the forearm under the drawn arm line, and
whatever was in or against the hand. The numbers were set on the stills of
the session 20261001_101621_left and every one of its 69 results was looked
at (2026-10-06): at `REACH` 0.9 a piece of the face beside the thumb came
through in one of them, at 0.5 in none. A new session's pictures have to be
looked at again: the rule is a cut by distance and brightness, not a
recognition of faces.

`write_images_pdf` puts the cut pictures of each grasp type on a page of
its own.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

REACH = 0.5        # radii from the skeleton a bright pixel may be and stay
CORE = 0.3         # radii from the skeleton every pixel stays
GATE = 0.7         # of the hand's own brightness: "bright"
LEVEL_PX = 6       # the hand's brightness is read this near the skeleton
MIN_SKELETON_PX = 150   # fewer drawn pixels than this: no hand to cut to
# The viewer's skeleton colours (`scripts/leap/camera_view.py` `COLOURS`,
# there in BGR), and the pixels taken as drawn in them: inclusive ranges of
# R, G and B (0 to 255), a band around the colour because the lines are
# anti-aliased.
SKELETON_COLOUR = {"left": (0, 220, 255), "right": (255, 60, 60)}
SKELETON_BAND = {"left": ((0, 89), (151, 255), (191, 255)),
                 "right": ((201, 255), (0, 109), (0, 109))}
# The viewer's yellow caption, the same way: drawn, not the hand.
CAPTION_BAND = ((201, 255), (201, 255), (0, 139))
DISTANCE_MASK = 5  # px: the distance transform's mask (OpenCV, L2)
CLOSE_PX = 5       # px: the side of the window the kept set is closed with

# What the stills of a session show in or against the hand, read by eye
# (repo-relative; keys: what, session, read_on, confirmed_by_operator,
# in_or_against_the_hand, in_the_other_hand, cannot_tell).
STILL_NOTES = "protocols/grasps_still_notes.json"


def in_band(rgb: np.ndarray, band) -> np.ndarray:
    """The pixels of `rgb` (H x W x 3, uint8) whose R, G and B each lie in
    the inclusive range `band` gives for it."""
    out = np.ones(rgb.shape[:2], dtype=bool)
    for k, (lo, hi) in enumerate(band):
        v = rgb[..., k].astype(int)
        out &= (v >= lo) & (v <= hi)
    return out


def skeleton_mask(rgb: np.ndarray, hand: str) -> np.ndarray:
    """The pixels of `rgb` (H x W x 3, uint8) drawn in the viewer's colour
    for a hand labelled `hand`: inside `SKELETON_BAND` (right for "right",
    left for anything else)."""
    return in_band(rgb, SKELETON_BAND["right" if hand == "right" else "left"])


def kept_mask(rgb: np.ndarray, hand: str = "left", reach: float = REACH,
              core: float = CORE, gate: float = GATE
              ) -> Optional[Tuple[np.ndarray, float]]:
    """(the pixels `hand_only` keeps, the hand's radius in pixels), or None
    when the still has no skeleton of that hand drawn on it."""
    import cv2

    sk = skeleton_mask(rgb, hand)
    if int(sk.sum()) < MIN_SKELETON_PX:
        return None
    ys, xs = np.nonzero(sk)
    cx, cy = float(np.median(xs)), float(np.median(ys))
    radius = float(np.median(np.hypot(xs - cx, ys - cy)))
    dist = cv2.distanceTransform((~sk).astype(np.uint8), cv2.DIST_L2,
                                 DISTANCE_MASK)
    grey = rgb.astype(float).mean(axis=2)
    # the skeleton and the yellow caption are drawn on the picture: they
    # say nothing about how bright the hand is
    drawn = sk | in_band(rgb, CAPTION_BAND)
    ring = (dist <= LEVEL_PX) & ~drawn
    level = float(np.median(grey[ring])) if ring.any() else 255.0
    keep = (dist <= core * radius) | ((dist <= reach * radius)
                                      & (grey >= gate * level))
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((CLOSE_PX, CLOSE_PX), np.uint8))
    _n, labels = cv2.connectedComponents(keep, connectivity=8)
    mine = np.unique(labels[sk])
    return np.isin(labels, mine[mine > 0]), radius


def hand_only(rgb: np.ndarray, hand: str = "left") -> Optional[np.ndarray]:
    """`rgb` with everything but the recorded hand black, cut to the hand;
    None when no skeleton of that hand is drawn on it (module docstring)."""
    found = kept_mask(rgb, hand)
    if found is None:
        return None
    keep, _radius = found
    out = rgb.copy()
    out[~keep] = 0
    ys, xs = np.nonzero(keep)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def has_hand(path: Path, hand: str = "left") -> bool:
    """A still that `hand_only` can cut: it opens, and the recorded hand's
    skeleton is drawn on it."""
    try:
        return kept_mask(read_rgb(path), hand) is not None
    except (OSError, ValueError):
        return False


def write_hand_image(src: Path, dest: Path, hand: str = "left") -> None:
    """The still `src` cut to the recorded hand, written as a PNG to `dest`
    (whatever its name: the build writes beside a file's place first). The
    same still gives the same bytes: nothing of the moment it is written
    goes into the file."""
    from PIL import Image

    cut = hand_only(read_rgb(src), hand)
    if cut is None:
        raise ValueError(f"{src}: no skeleton of the {hand} hand is drawn "
                         "on the still")
    with open(dest, "wb") as f:
        Image.fromarray(cut).save(f, format="PNG", optimize=False)


# The page of `write_images_pdf`: A4 in points, its margin, three pictures
# to a row with GAP between them, and the fonts of its texts: (font, size,
# the step from one line to the next).
PAGE_W, PAGE_H = 210 * 72 / 25.4, 297 * 72 / 25.4   # A4, as reportlab's
MARGIN, PER_ROW, GAP = 36.0, 3, 14.0
TEXT_W = PAGE_W - 2 * MARGIN                  # the usable width of a line
CELL_W = (TEXT_W - GAP * (PER_ROW - 1)) / PER_ROW
TITLE_FONT = ("Helvetica-Bold", 13.0, 17.0)   # the grasp's name
LABEL_FONT = ("Helvetica", 10.0, 13.0)        # its label, paper and figure
LINE_FONT = ("Helvetica", 9.0, 12.0)          # the page's other lines
CAPTION_FONT = ("Helvetica", 7.5, 9.0)        # under each picture
FOOTER_FONT = ("Helvetica", 7.5, 9.0)
FOOTER_Y = MARGIN - 12.0


def text_width(text: str, font: str, size: float) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    return float(stringWidth(text, font, size))


def wrap_text(text: str, font: str, size: float, width: float) -> List[str]:
    """`text` broken into lines no wider than `width` points in `font`:
    at spaces, and a word wider than a line on its own after its last
    underscore, slash or hyphen that fits (a take's name breaks before
    "take<N>"), else at the last character that fits, so nothing is ever
    drawn past `width`."""
    out: List[str] = []
    line = ""
    for word in str(text).split():
        trial = f"{line} {word}" if line else word
        if text_width(trial, font, size) <= width:
            line = trial
            continue
        if line:
            out.append(line)
        while text_width(word, font, size) > width:
            k = len(word) - 1
            while k > 1 and text_width(word[:k], font, size) > width:
                k -= 1
            cut = max(word.rfind(sep, 0, k) for sep in "_/\\-") + 1
            k = cut if cut > 0 else k
            out.append(word[:k])
            word = word[k:]
        line = word
    return out + ([line] if line else [])


def title_lines(title: str, label: str = "", lines: Sequence[str] = (),
                width: float = TEXT_W, source: str = ""
                ) -> List[Tuple[str, float, str]]:
    """The head of a page as (font, size, text) lines, each no wider than
    `width`: the grasp's name (`title`) in TITLE_FONT; then in LABEL_FONT
    its label with the paper and figure (`source`) in brackets after it
    when that fits on one line, else the label and on the next line the
    source; then the page's `lines` in LINE_FONT. Every one is wrapped at
    word boundaries (`wrap_text`)."""
    out: List[Tuple[str, float, str]] = []

    def add(text: str, spec) -> None:
        font, size, _step = spec
        out.extend((font, size, x) for x in wrap_text(text, font, size,
                                                      width))
    add(title, TITLE_FONT)
    both = f"{label} ({source})" if label and source else (label or source)
    font, size, _step = LABEL_FONT
    if both and text_width(both, font, size) <= width:
        out.append((font, size, both))
    else:
        add(label, LABEL_FONT)
        add(source, LABEL_FONT)
    for line in lines:
        add(line, LINE_FONT)
    return out


def page_layout(page: dict) -> dict:
    """Where everything of a page of `write_images_pdf` goes, in points
    from the bottom left: "head" [(font, size, text, x, y)], "cells"
    [(x, top, width, height)] one per picture, "captions" [(font, size,
    text, x, y)], "footer_y". Nothing in it depends on the pictures'
    sizes: a picture is fitted into its cell when it is drawn."""
    step = {TITLE_FONT[:2]: TITLE_FONT[2], LABEL_FONT[:2]: LABEL_FONT[2],
            LINE_FONT[:2]: LINE_FONT[2]}
    head = []
    y = PAGE_H - MARGIN
    for font, size, text in title_lines(
            page.get("title", ""), page.get("label", ""),
            page.get("lines", []), TEXT_W, page.get("source", "")):
        head.append((font, size, text, MARGIN, y - size))
        y -= step[(font, size)]
    y -= 8.0
    font, size, cstep = CAPTION_FONT
    captions_of = [[x for line in caption
                    for x in wrap_text(line, font, size, CELL_W)]
                   for caption, _still, _hand in page["pictures"]]
    n = max([len(c) for c in captions_of] + [1])
    caption_h = 10.0 + cstep * n + 6.0      # under a picture, to the next row
    rows = max(1, (len(page["pictures"]) + PER_ROW - 1) // PER_ROW)
    cell_h = min(250.0, (y - MARGIN - 30.0) / rows - caption_h)
    cells, captions = [], []
    for k, lines in enumerate(captions_of):
        col, row = k % PER_ROW, k // PER_ROW
        x0 = MARGIN + col * (CELL_W + GAP)
        top = y - row * (cell_h + caption_h)
        cells.append((x0, top, CELL_W, cell_h))
        for j, line in enumerate(lines):
            captions.append((font, size, line, x0,
                             top - cell_h - 10.0 - cstep * j))
    return {"head": head, "cells": cells, "captions": captions,
            "footer_y": FOOTER_Y}


def write_images_pdf(dest: Path, pages: Sequence[dict], footer: str) -> None:
    """One A4 page per grasp type with its cut pictures side by side.

    A page is {"title": the grasp's name, "label": its label, "source": its
    paper and figure, "lines": [str], "pictures": [(caption lines, still
    path, hand)]} ("label", "source" and "lines" may be left out); its head
    and captions are laid out by `page_layout`, every line wrapped to its
    width, so no text runs off the page or into the next picture. The
    pictures are cut here from the stills again, so the PDF does not depend
    on the PNG files being written first. Three to a row, each as large as
    its cell allows; a small picture is drawn enlarged, not resampled, so
    its pixels are the camera's. invariant=1 and no time in it: the same
    pages give the same bytes.
    """
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(dest), pagesize=(PAGE_W, PAGE_H), invariant=1)
    c.setTitle("Infrared pictures of the grasps, cut to the recorded hand")
    for page in pages:
        laid = page_layout(page)
        c.setFillColorRGB(0, 0, 0)
        for font, size, text, x, y in laid["head"]:
            c.setFont(font, size)
            c.drawString(x, y, text)
        for (x0, top, w_cell, h_cell), (_caption, still, hand) in zip(
                laid["cells"], page["pictures"]):
            cut = hand_only(read_rgb(Path(still)), hand)
            if cut is not None:
                h, w = cut.shape[:2]
                s = min(w_cell / w, h_cell / h)
                c.drawImage(ImageReader(Image.fromarray(cut)), x0,
                            top - h * s, width=w * s, height=h * s)
        for font, size, text, x, y in laid["captions"]:
            c.setFont(font, size)
            c.drawString(x, y, text)
        font, size, _step = FOOTER_FONT
        c.setFont(font, size)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        for k, line in enumerate(wrap_text(footer, font, size, TEXT_W)):
            c.drawString(MARGIN, laid["footer_y"] - FOOTER_FONT[2] * k, line)
        c.showPage()
    c.save()


def load_still_notes(repo_root: Path, session_name: Optional[str] = None
                     ) -> dict:
    """The still notes (`STILL_NOTES` under `repo_root`) as a dict; {} when
    the file is missing, unreadable or not a JSON object. A ValueError when
    `session_name` is given and the notes say they are of another session:
    notes read off one session's stills say nothing about another's."""
    try:
        with open(Path(repo_root) / STILL_NOTES, encoding="utf-8") as f:
            notes = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(notes, dict):
        return {}
    if (session_name is not None and "session" in notes
            and str(notes["session"]) != session_name):
        raise ValueError(f"{STILL_NOTES} is of the session "
                         f"{notes['session']!r}, not {session_name!r}")
    return notes


def looked_at_note(notes: dict, name: str) -> Tuple[str, str]:
    """(what the picture of the take `name` shows in or against the
    recorded hand, what it shows in the other hand), each "" when the notes
    say nothing: `notes` is a session's `still_notes.json`."""
    def of(key: str) -> str:
        table = notes.get(key) if isinstance(notes, dict) else None
        return str((table or {}).get(name, "")) if isinstance(table, dict) \
            else ""
    return of("in_or_against_the_hand"), of("in_the_other_hand")


def unclear(notes: dict) -> List[str]:
    """The takes whose picture does not let one tell (`still_notes.json`)."""
    table = notes.get("cannot_tell") if isinstance(notes, dict) else None
    return [str(x) for x in table] if isinstance(table, list) else []


def read_by_eye(notes: dict) -> List[str]:
    """The takes whose picture was read by eye (`still_notes.json`, "read");
    [] when the notes have no such list. A take not in it has not been
    read: the notes say nothing about its picture."""
    table = notes.get("read") if isinstance(notes, dict) else None
    return [str(x) for x in table] if isinstance(table, list) else []
