"""leap_hand.grasp_images: cutting a still to the recorded hand, the PNG and
PDF it is written to, and the still notes read by eye.

The stills here are built in the test: a dark picture with a bright hand
(a disc and five finger bars) carrying the viewer's skeleton colour, a
bright "face" far from it and a yellow caption, so what has to be kept and
what has to go are known pixel by pixel. One test reads a real still of the
2026-10-01 grasp session when it is on disk.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from leap_hand import grasp_images as gi

REPO = Path(__file__).resolve().parents[1]
SESSION_NAME = "20261001_101621_left"
SESSION = REPO / "recordings" / "protocol" / "grasps" / SESSION_NAME
REAL_STILL = SESSION / "stills" / "p1_cylindrical_left_take1_20261001_101712.png"

LEFT = (0, 220, 255)            # the viewer's colour for a hand labelled left
RIGHT = (255, 60, 60)           # ... and right
YELLOW = (255, 255, 0)          # the caption
BACKGROUND = 12                 # dark, but not black: kept pixels stay > 0
HAND_GREY = 200
FACE_GREY = 235


def blank(size=300):
    return np.full((size, size, 3), BACKGROUND, np.uint8)


def draw_hand(rgb, cx, cy, colour):
    """A bright hand (a palm disc, four finger bars, a thumb) with its
    skeleton drawn anti-aliased in `colour`; returns the hand's own mask."""
    hand = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(hand, (cx, cy), 42, 1, -1)
    tips = []
    for dx in (-30, -10, 10, 30):
        cv2.rectangle(hand, (cx + dx - 6, cy - 110), (cx + dx + 6, cy), 1, -1)
        tips.append((cx + dx, cy - 104))
    cv2.line(hand, (cx - 20, cy), (cx - 70, cy - 50), 1, 14)
    tips.append((cx - 66, cy - 46))
    rgb[hand.astype(bool)] = HAND_GREY
    for tip in tips:
        cv2.line(rgb, (cx, cy), tip, colour, 2, cv2.LINE_AA)
    cv2.line(rgb, (cx, cy), (cx, cy + 36), colour, 2, cv2.LINE_AA)
    return hand.astype(bool)


def still(hand="left"):
    """(picture, the hand's mask, the face's mask, the caption's mask)."""
    rgb = blank()
    hand_mask = draw_hand(rgb, 110, 170, LEFT if hand == "left" else RIGHT)
    face = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(face, (255, 45), 26, 1, -1)
    face = face.astype(bool)
    rgb[face] = FACE_GREY
    before = rgb.copy()
    cv2.putText(rgb, "p1_test take 1", (4, 290), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, YELLOW, 1, cv2.LINE_AA)
    caption = np.any(rgb != before, axis=2)
    return rgb, hand_mask, face, caption


def write_png(rgb, path):
    from PIL import Image
    Image.fromarray(rgb).save(path)
    return path


# --- the cut -------------------------------------------------------------------------
def test_skeleton_mask_finds_the_skeleton_and_not_the_caption():
    rgb, _hand, _face, caption = still()
    sk = gi.skeleton_mask(rgb, "left")
    exact = np.all(rgb == LEFT, axis=2)
    assert exact.sum() >= gi.MIN_SKELETON_PX
    assert sk[exact].all()
    assert sk.sum() >= gi.MIN_SKELETON_PX
    assert not (sk & caption).any()
    assert not sk[np.all(rgb == YELLOW, axis=2)].any()
    assert not gi.skeleton_mask(rgb, "right").any()


def test_kept_mask_needs_min_skeleton_px_drawn_pixels():
    assert gi.kept_mask(blank(), "left") is None
    rgb = blank()
    rgb[100:110, 100:115] = LEFT                    # 150 pixels
    assert int(gi.skeleton_mask(rgb, "left").sum()) == gi.MIN_SKELETON_PX
    assert gi.kept_mask(rgb, "left") is not None
    rgb[100, 100] = BACKGROUND                      # 149
    assert gi.kept_mask(rgb, "left") is None
    assert gi.hand_only(rgb, "left") is None


def test_hand_only_keeps_the_hand_and_blackens_the_face():
    rgb, hand, face, caption = still()
    found = gi.kept_mask(rgb, "left")
    assert found is not None
    keep, radius = found
    out = gi.hand_only(rgb, "left")
    assert out is not None and out.dtype == np.uint8 and out.shape[2] == 3

    # the cut: the kept pixels as they were, black elsewhere, cut to the
    # kept pixels' bounding box
    ys, xs = np.nonzero(keep)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    assert out.shape[:2] == (y1 - y0 + 1, x1 - x0 + 1)
    want = np.where(keep[..., None], rgb, 0)[y0:y1 + 1, x0:x1 + 1]
    assert np.array_equal(out, want)
    lit = np.any(out > 0, axis=2)
    assert lit[0].any() and lit[-1].any()
    assert lit[:, 0].any() and lit[:, -1].any()
    assert out.shape[0] < rgb.shape[0] and out.shape[1] < rgb.shape[1]

    # measured again here: every pixel left lit lies within REACH radii of
    # a skeleton pixel (plus the 5 x 5 closing's reach, 2 px diagonally)
    sk = gi.skeleton_mask(rgb, "left")
    sy, sx = np.nonzero(sk)
    my, mx = np.median(sy), np.median(sx)
    r = float(np.median(np.hypot(sx - mx, sy - my)))
    assert r == pytest.approx(radius)
    dist = cv2.distanceTransform((~sk).astype(np.uint8), cv2.DIST_L2, 5)
    ly, lx = np.nonzero(lit)
    assert dist[ly + y0, lx + x0].max() <= gi.REACH * r + 2 * np.sqrt(2)

    # the face and the caption are gone, the skeleton and the hand near it
    # are kept
    assert not keep[face].any() and not keep[caption].any()
    assert not np.any(np.all(out == FACE_GREY, axis=2))
    assert not np.any(np.all(out == YELLOW, axis=2))
    assert keep[sk].all()
    near = hand & (dist <= gi.REACH * r)
    assert near.sum() > 1000 and keep[near].all()
    assert keep[hand & (dist <= gi.CORE * r)].all()


def test_the_right_hands_colour_is_used_for_hand_right():
    rgb = blank(320)
    left = draw_hand(rgb, 90, 180, LEFT)
    right = draw_hand(rgb, 240, 180, RIGHT)
    assert not (left & right).any()
    sk_r = gi.skeleton_mask(rgb, "right")
    assert sk_r[np.all(rgb == RIGHT, axis=2)].all()
    assert not sk_r[np.all(rgb == LEFT, axis=2)].any()
    keep_r, _ = gi.kept_mask(rgb, "right")
    keep_l, _ = gi.kept_mask(rgb, "left")
    assert keep_r[right].any() and not keep_r[left].any()
    assert keep_l[left].any() and not keep_l[right].any()
    out = gi.hand_only(rgb, "right")
    assert np.any(np.all(out == RIGHT, axis=2))
    assert not np.any(np.all(out == LEFT, axis=2))
    only_right, _h, _f, _c = still("right")
    assert gi.hand_only(only_right, "left") is None
    assert gi.hand_only(only_right, "right") is not None


# --- the files -----------------------------------------------------------------------
def test_write_hand_image_is_deterministic_and_refuses_a_still_without_hand(
        tmp_path):
    from PIL import Image

    src = write_png(still()[0], tmp_path / "still.png")
    a, b = tmp_path / "a.png", tmp_path / "b.part"
    gi.write_hand_image(src, a)
    gi.write_hand_image(src, b)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(a) as im:
        assert np.array_equal(np.asarray(im.convert("RGB")),
                              gi.hand_only(gi.read_rgb(src), "left"))
    empty = write_png(blank(), tmp_path / "empty.png")
    with pytest.raises(ValueError):
        gi.write_hand_image(empty, tmp_path / "c.png")
    assert not (tmp_path / "c.png").exists()


def test_has_hand(tmp_path):
    assert gi.has_hand(write_png(still()[0], tmp_path / "s.png"))
    assert not gi.has_hand(write_png(blank(), tmp_path / "e.png"))
    assert not gi.has_hand(write_png(still("right")[0], tmp_path / "r.png"),
                           "left")
    assert gi.has_hand(tmp_path / "r.png", "right")
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a picture")
    assert not gi.has_hand(junk)
    assert not gi.has_hand(tmp_path / "missing.png")


def test_write_images_pdf_one_page_per_entry_and_deterministic(tmp_path):
    from pypdf import PdfReader

    left = write_png(still()[0], tmp_path / "left.png")
    other = still()[0]
    other[160:180, 100:104] = 180               # another take, other pixels
    left2 = write_png(other, tmp_path / "left2.png")
    right = write_png(still("right")[0], tmp_path / "right.png")
    empty = write_png(blank(), tmp_path / "empty.png")
    pages = [
        {"title": "Grasp one", "lines": ["two recordings"],
         "pictures": [(["take 1"], left, "left"),
                      (["take 2"], left2, "left")]},
        {"title": "Grasp two", "lines": [],
         "pictures": [(["take 1"], right, "right")]},
        {"title": "Grasp three", "lines": ["a still without the hand"],
         "pictures": [(["take 1", "no skeleton"], empty, "left")]},
    ]
    a, b = tmp_path / "a.pdf", tmp_path / "another.pdf"
    gi.write_images_pdf(a, pages, "footer of the test")
    gi.write_images_pdf(b, pages, "footer of the test")
    assert a.read_bytes() == b.read_bytes()
    reader = PdfReader(str(a))
    assert len(reader.pages) == len(pages)
    for page, want in zip(reader.pages, pages):
        text = page.extract_text()
        assert want["title"] in text and "footer of the test" in text
    images = [len(p.images) for p in reader.pages]
    assert images == [2, 1, 0]


def _words(stem: str, n_chars: int) -> str:
    """Distinct words joined by spaces, `n_chars` long (the last one cut
    short, never ending in a space)."""
    text, k = "", 0
    while len(text) < n_chars:
        text += ("" if not text else " ") + f"{stem}{k:03d}"
        k += 1
    text = text[:n_chars]
    return text if not text.endswith(" ") else text[:-1] + "x"


def test_a_long_title_and_label_are_wrapped_whole_and_fit(tmp_path):
    """A 200-character name and a 150-character label: every laid-out line
    is no wider than the page's text width, the PDF holds both whole, and
    the lowest caption stays above the footer."""
    from pypdf import PdfReader
    from reportlab.pdfbase.pdfmetrics import stringWidth

    title, label = _words("name", 200), _words("label", 150)
    assert (len(title), len(label)) == (200, 150)
    source = "p9 Some Author et al. 2020, Figure 12, panel 3; Table 4"
    lines = gi.title_lines(title, label, ["Shape asked for: " + label],
                           gi.TEXT_W, source)
    assert all(stringWidth(t, f, s) <= gi.TEXT_W for f, s, t in lines)
    assert lines[0][:2] == ("Helvetica-Bold", 13.0)
    assert {(f, s) for f, s, _t in lines} == {
        ("Helvetica-Bold", 13.0), ("Helvetica", 10.0), ("Helvetica", 9.0)}
    joined = " ".join(t for _f, _s, t in lines)
    assert title in joined and label in joined and source in joined
    # one word wider than the line is broken, never drawn past it
    one = gi.wrap_text("x" * 400, "Helvetica-Bold", 13.0, 100.0)
    assert "".join(one) == "x" * 400
    assert all(stringWidth(x, "Helvetica-Bold", 13.0) <= 100.0 for x in one)
    # a take's name too long for its cell breaks at an underscore
    assert gi.wrap_text("p2_s09_directional_power_side_and_tips_left_take1",
                        "Helvetica", 7.5, gi.CELL_W) == [
        "p2_s09_directional_power_side_and_tips_left_", "take1"]

    picture = write_png(still()[0], tmp_path / "left.png")
    page = {"title": title, "label": label, "source": source,
            "lines": ["a line " * 30],
            "pictures": [([f"{_words('cap', 60)}", "take 1, 2026-10-01"],
                          picture, "left")] * 5}
    pdf = tmp_path / "long.pdf"
    gi.write_images_pdf(pdf, [page], "footer of the test")
    text = " ".join(PdfReader(str(pdf)).pages[0].extract_text().split())
    assert title in text and label in text and source in text
    laid = gi.page_layout(page)
    for font, size, words, x, _y in laid["head"]:
        assert x + stringWidth(words, font, size) <= gi.PAGE_W - gi.MARGIN
    for font, size, words, x, _y in laid["captions"]:
        assert stringWidth(words, font, size) <= gi.CELL_W
    lowest = min(y for *_rest, y in laid["captions"])
    assert lowest > laid["footer_y"] + gi.CAPTION_FONT[1]
    assert min(y for *_rest, y in laid["head"]) > max(
        top for _x, top, _w, _h in laid["cells"])
    assert all(h > 50.0 for *_r, h in laid["cells"])


GRASPS = REPO / "protocols" / "grasps.json"
ITEMS = (json.loads(GRASPS.read_text(encoding="utf-8"))["items"]
         if GRASPS.is_file() else [])


@pytest.mark.skipif(not ITEMS, reason="protocols/grasps.json is absent")
@pytest.mark.parametrize("item", ITEMS, ids=[str(i.get("id")) for i in ITEMS])
def test_every_grasps_page_head_fits_and_its_captions_stay_on_the_page(item):
    """The head of each real grasp's page, as the build lays it out
    (`leap_hand.grasp_handin.image_pdf_pages`: the name, the label with
    the paper and figure, the shape asked for, the notes read by eye, here
    the longest kind: in the other hand, not in the picture), and the most
    pictures a grasp has had (5), each with the longest caption."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    where = ", ".join(x for x in (str(item.get("source") or ""),
                                  str(item.get("figure") or "")) if x)
    notes = ["What was seen, read by eye on 2026-10-06 from the recorder's "
             "uncut stills; not confirmed by the operator:"] + [
        f"take {n}: in the other hand, which is not in the picture below: "
        "something held, not identified" for n in range(1, 6)]
    lines = gi.title_lines(str(item["id"]), str(item.get("label") or ""),
                           [f"Shape asked for: {item.get('shape', '')}"]
                           + notes, gi.TEXT_W, where)
    for font, size, text in lines:
        assert stringWidth(text, font, size) <= gi.TEXT_W, text
    stem = f"{item['id']}_left_take5"
    page = {"title": str(item["id"]), "label": str(item.get("label") or ""),
            "source": where,
            "lines": [f"Shape asked for: {item.get('shape', '')}"] + notes,
            "pictures": [([stem, "take 5, 2026-10-02",
                           "by eye, uncut still: something in the other "
                           "hand (not in this picture)"],
                          Path("unused.png"), "left")] * 5}
    laid = gi.page_layout(page)
    for font, size, text, x, _y in laid["head"] + laid["captions"]:
        assert x + stringWidth(text, font, size) <= gi.PAGE_W - gi.MARGIN
    for font, size, text, _x, _y in laid["captions"]:
        assert stringWidth(text, font, size) <= gi.CELL_W, text
    assert min(y for *_r, y in laid["captions"]) > (laid["footer_y"]
                                                    + gi.CAPTION_FONT[1])
    assert all(h > 100.0 for *_r, h in laid["cells"])


# --- the still notes -----------------------------------------------------------------
NOTES = {
    "what": "test notes", "session": SESSION_NAME, "read_on": "2026-10-06",
    "confirmed_by_operator": False,
    "in_or_against_the_hand": {"p1_palmar_left_take1": "a white sheet or card"},
    "in_the_other_hand": {"p3_circular_power_left_take1":
                          "something held, not identified"},
    "cannot_tell": ["p1_cylindrical_left_take1", "p1_lateral_left_take2"],
}


def test_looked_at_note_and_unclear_read_the_notes():
    assert gi.looked_at_note(NOTES, "p1_palmar_left_take1") == \
        ("a white sheet or card", "")
    assert gi.looked_at_note(NOTES, "p3_circular_power_left_take1") == \
        ("", "something held, not identified")
    assert gi.looked_at_note(NOTES, "p1_hook_left_take1") == ("", "")
    assert gi.looked_at_note({}, "p1_palmar_left_take1") == ("", "")
    assert gi.looked_at_note(None, "p1_palmar_left_take1") == ("", "")
    assert gi.looked_at_note({"in_or_against_the_hand": ["x"]}, "x") == \
        ("", "")
    assert gi.unclear(NOTES) == ["p1_cylindrical_left_take1",
                                 "p1_lateral_left_take2"]
    assert gi.unclear({}) == []
    assert gi.unclear({"cannot_tell": "p1"}) == []
    assert gi.unclear(None) == []


def test_read_by_eye_is_the_notes_read_list_and_nothing_else():
    """"read" lists the pictures looked at; without it (or with something
    that is not a list) no picture is said to be read."""
    read = ["p1_cylindrical_left_take1", "p1_palmar_left_take1",
            "p1_hook_left_take1"]
    assert gi.read_by_eye({**NOTES, "read": read}) == read
    assert gi.read_by_eye({**NOTES, "read": [3, "x"]}) == ["3", "x"]
    assert gi.read_by_eye(NOTES) == []                         # absent
    assert gi.read_by_eye({**NOTES, "read": "p1_hook_left_take1"}) == []
    assert gi.read_by_eye({**NOTES, "read": {"p1_hook_left_take1": 1}}) == []
    assert gi.read_by_eye({**NOTES, "read": None}) == []
    assert gi.read_by_eye({}) == []
    assert gi.read_by_eye(None) == []
    assert gi.read_by_eye(["read"]) == []


def test_load_still_notes(tmp_path):
    assert gi.load_still_notes(tmp_path) == {}
    path = tmp_path / gi.STILL_NOTES
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(NOTES), encoding="utf-8")
    assert gi.load_still_notes(tmp_path) == NOTES
    assert gi.load_still_notes(tmp_path, SESSION_NAME) == NOTES
    with pytest.raises(ValueError):
        gi.load_still_notes(tmp_path, "20261002_090000_left")
    no_session = {k: v for k, v in NOTES.items() if k != "session"}
    path.write_text(json.dumps(no_session), encoding="utf-8")
    assert gi.load_still_notes(tmp_path, "any session") == no_session
    path.write_text("{not json", encoding="utf-8")
    assert gi.load_still_notes(tmp_path) == {}
    path.write_text("[1, 2]", encoding="utf-8")
    assert gi.load_still_notes(tmp_path) == {}
    path.write_bytes(b"\xff\xfe\x00bad")
    assert gi.load_still_notes(tmp_path) == {}


def test_the_repos_still_notes_are_of_the_session_and_say_how_they_were_read():
    notes = gi.load_still_notes(REPO, SESSION_NAME)
    assert notes["session"] == SESSION_NAME
    # the day or days of the reading: ISO dates joined by " and ", in order,
    # none before the first reading (2026-10-06); the 46 pictures recorded
    # on 7 and 8 October were read on 2026-10-08
    days = notes["read_on"].split(" and ")
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in days), days
    assert days == sorted(set(days)) and days[0] >= "2026-10-06", days
    assert notes["confirmed_by_operator"] is False
    assert "by eye" in notes["what"] and "2026-10-06" in notes["what"]
    assert all(d in notes["what"] for d in days)
    assert "not confirmed by the operator" in notes["what"]
    against, other = notes["in_or_against_the_hand"], notes["in_the_other_hand"]
    cannot = notes["cannot_tell"]
    stems = list(against) + list(other) + list(cannot)
    assert len(stems) == len(set(stems))
    assert gi.looked_at_note(notes, "p1_tip_left_take1") == \
        ("a thin rod like a pen", "")
    assert gi.unclear(notes) == cannot
    if not (SESSION / "session.json").is_file():
        pytest.skip("the 2026-10-01 grasp session is not on disk")
    session = json.loads((SESSION / "session.json").read_text(
        encoding="utf-8"))
    accepted = {f"{t['item']}_{session['hand']}_take{t['take']}"
                for t in session["takes"] if t.get("accepted")}
    assert set(stems) <= accepted
    # the first reading (2026-10-06), of the 69 takes of 1 and 2 October,
    # stands as it was: 4 in or against the hand, 2 in the other, 15 that
    # do not let one tell
    first = {f"{t['item']}_{session['hand']}_take{t['take']}"
             for t in session["takes"] if t.get("accepted")
             and t["name"].rsplit("_", 2)[-2] in ("20261001", "20261002")}
    assert len(first) == 69
    assert (len(set(against) & first), len(set(other) & first),
            len(set(cannot) & first)) == (4, 2, 15)
    # what was read: every accepted take of the session, each once, in the
    # session's order
    read = gi.read_by_eye(notes)
    assert len(read) == len(set(read))
    assert set(stems) <= set(read) == accepted
    order = [f"{t['item']}_{session['hand']}_take{t['take']}"
             for t in session["takes"] if t.get("accepted")]
    assert read == [s for s in order if s in set(read)]
    assert "read lists every take whose picture was looked at" in \
        notes["what"]
    assert "not in read has not been read" in notes["what"]


# --- a real still --------------------------------------------------------------------
@pytest.mark.skipif(not REAL_STILL.is_file(),
                    reason="the 2026-10-01 grasp session is not on disk")
def test_a_real_still_has_the_hand_and_is_cut_smaller():
    assert gi.has_hand(REAL_STILL)
    rgb = gi.read_rgb(REAL_STILL)
    out = gi.hand_only(rgb)
    assert out is not None and out.size > 0
    assert np.any(out > 0)
    assert out.shape[0] <= rgb.shape[0] and out.shape[1] <= rgb.shape[1]
    assert out.shape[0] * out.shape[1] < rgb.shape[0] * rgb.shape[1]
