"""One window for a visitor: hand detection from the webcam, hand tracking
from the Ultraleap 3D infrared sensor, and the hand the StretchSense glove
reports, side by side, with their numbers underneath. Nothing is recorded.

    TOP LEFT      the laptop's webcam picture with the 21 points MediaPipe
                  finds on each hand (cam_hand.landmarks, the August path)
    TOP MIDDLE    the Ultraleap Stereo IR 170's infrared picture with the
                  skeleton the tracking service fitted (the same read-only
                  client as scripts/leap/camera_view.py)
    TOP RIGHT     the glove's hand, drawn looking at the palm, from XR Train's
                  OSC stream on port 9002 (the stream every glove recorder in
                  this repo reads: xr_hand.receiver -> parser -> keypoints21)
    BOTTOM LEFT   x, y, z in centimetres from the sensor for the palm and the
                  five fingertips of each hand the sensor tracks
    BOTTOM RIGHT  the same six points from the glove, centimetres from the
                  wrist (the glove measures the hand's shape, not its place)

The look is a quiet dark one: three text sizes, grey labels, white numbers,
green only for what is live. A table shows rows only for the hands it has,
so with one glove on there are 18 numbers on that side, not 36 dashes.

Usage:
  python scripts/minister_demo.py                  # all three sources
  python scripts/minister_demo.py --fullscreen     # the demo itself
  python scripts/minister_demo.py --no-glove       # without XR Train
  python scripts/minister_demo.py --mock --no-window --frames 20 --snapshot x.png

Keys: q or Esc closes, f toggles full screen.

A source that cannot be opened does not stop the window: its panel says so
in words ("sensor not connected", "glove stream not available") and the
other panels keep going, so a loose cable on the day shows one broken panel,
not a crash.
"""
import argparse
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from cam_hand.demo_view import MM_PER_PX, draw_hand as draw_palm_view
from cam_hand.fusion import palm_basis

REPO = Path(__file__).resolve().parents[1]

# ---- layout (pixels of the drawn canvas, 1920 x 1080; the window scales it)
CANVAS_W, CANVAS_H = 1920, 1080
MARGIN = 16
HEADER = 90
PANEL_Y = HEADER
PANEL_H = 400
WEB_W = 533                     # 4:3 at PANEL_H
IR_W = 400                      # the sensor image is square
GLOVE_W = 600                   # two palm-view hands side by side
WEB_X = MARGIN
IR_X = WEB_X + WEB_W + MARGIN
GLOVE_X = IR_X + IR_W + MARGIN
TABLE_Y = PANEL_Y + PANEL_H + MARGIN
TABLE_H = CANVAS_H - TABLE_Y - 30
TABLE_W = 936
SENSOR_TABLE_X = MARGIN
GLOVE_TABLE_X = SENSOR_TABLE_X + TABLE_W + MARGIN
TOP_BAND = 80                   # the title strip inside a picture card
BOTTOM_BAND = 46                # the status strip

# ---- colours (BGR), a dark palette with one accent
BG = (10, 10, 10)
CARD = (30, 28, 28)
WHITE = (255, 255, 255)
VALUE = (240, 240, 240)         # the live numbers
LABEL = (147, 142, 142)         # captions and column names
GREY = LABEL
GREEN = (88, 209, 48)           # the skeletons and the "live" dot
LIME = GREEN                    # older name, kept for the tests
AMBER = (10, 214, 255)
RED = (58, 69, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

SENSOR_PX = 384                 # the SIR170 image is 384 x 384
IR_SCALE = IR_W / SENSOR_PX
BASELINE_HALF_MM = 32.0         # lenses 64 mm apart; the left lens sits at x = +32 mm (see camera_view.py)
LEFT_CAMERA = 1                 # eLeapPerspectiveType_stereo_left
FRESH_S = 0.5                   # a stream older than this is drawn as no hand

# MediaPipe's 21-point topology
MP_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)]
MP_TIPS = {"thumb tip": 4, "index tip": 8, "middle tip": 12, "ring tip": 16, "pinky tip": 20}
MP_KNUCKLES = (5, 9, 13, 17)
MP_PARENT = [None, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

# the joint frames drawn on the hands, as in Cobos 2009 figure 1 and docs/joint_frames.md:
# z along the bone toward the wrist, y out of the back of the hand, x = y cross z
AXIS_COLOURS = {"x": (58, 69, 255), "y": (10, 214, 255), "z": (255, 159, 10)}     # red, yellow, blue
AXIS_MM = 12.0                  # a joint's axes
WRIST_AXIS_MM = 28.0            # the wrist frame, drawn bigger

# the columns of the numbers tables, in order: the key in a points dict and the word shown
POINT_NAMES = ["palm", "thumb tip", "index tip", "middle tip", "ring tip", "pinky tip"]
COLUMN_WORDS = ["palm", "thumb", "index", "middle", "ring", "pinky"]
SIDES = ("left", "right")
# the bend rows: which joint of each finger (thumb, index, middle, ring, pinky) each row reports
BEND_ROWS = {"knuckle": (1, 5, 9, 13, 17), "middle": (2, 6, 10, 14, 18), "end": (3, 7, 11, 15, 19)}

WIN = "Hand tracking - webcam, 3D sensor and glove"


# ---------------------------------------------------------------- drawing helpers
def put(img, text, org, scale=0.6, colour=WHITE, thick=1):
    """Text. No outline: OpenCV 5 draws a thicker stroke as a wider font, so an
    outline pokes out past the text (see cam_hand.demo_view.put)."""
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, colour, thick, cv2.LINE_AA)


def put_right(img, text, right_x, y, scale, colour, thick=1):
    w = cv2.getTextSize(text, FONT, scale, thick)[0][0]
    cv2.putText(img, text, (int(right_x - w), int(y)), FONT, scale, colour, thick, cv2.LINE_AA)


def put_centre(img, text, cx, y, scale, colour, thick=1):
    w = cv2.getTextSize(text, FONT, scale, thick)[0][0]
    cv2.putText(img, text, (int(cx - w / 2), int(y)), FONT, scale, colour, thick, cv2.LINE_AA)


def darken(img, y0, y1, keep=0.42):
    """Dim a horizontal strip of a picture so text reads on it."""
    strip = img[y0:y1]
    strip[:] = (strip * keep).astype(np.uint8)


def card(width, height):
    return np.full((height, width, 3), CARD, np.uint8)


def title_strip(panel, title, caption, on_picture):
    """The card's name and a short caption, top-left, over a dimmed strip on a picture."""
    if on_picture:
        darken(panel, 0, TOP_BAND)
    put(panel, title, (24, 38), 0.8, WHITE, 2)
    put(panel, caption, (24, 64), 0.5, LABEL, 1)


def status_strip(panel, text, colour, on_picture):
    """A dot and a few words at the bottom of a card."""
    h = panel.shape[0]
    if on_picture:
        darken(panel, h - BOTTOM_BAND, h)
    cv2.circle(panel, (30, h - 20), 6, colour, -1, cv2.LINE_AA)
    put(panel, text, (46, h - 14), 0.58, WHITE, 1)


def draw_mp_hand(frame, pts, colour=GREEN):
    """Draw MediaPipe's 21 image-space points (x_px, y_px, ...) on `frame`."""
    px = [(int(round(p[0])), int(round(p[1]))) for p in pts]
    for a, b in MP_BONES:
        cv2.line(frame, px[a], px[b], colour, 3, cv2.LINE_AA)
    for p in px:
        cv2.circle(frame, p, 5, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 5, colour, 1, cv2.LINE_AA)
    return frame


def draw_leap_hand(frame, hand, to_px, colour=GREEN):
    """Draw one tracked hand (wrist, elbow, five digit chains in mm) with `to_px`."""
    w = to_px(hand["wrist"])
    e = to_px(hand["elbow"])
    if w and e:
        cv2.line(frame, e, w, colour, 3, cv2.LINE_AA)
    for chain in hand["chains"]:
        pts = [to_px(p) for p in chain]
        if w and pts[0]:
            cv2.line(frame, w, pts[0], colour, 2, cv2.LINE_AA)
        for a, b in zip(pts, pts[1:]):
            if a and b:
                cv2.line(frame, a, b, colour, 3, cv2.LINE_AA)
        for p in pts:
            if p:
                cv2.circle(frame, p, 4, WHITE, -1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------- joint frames (the x, y, z at every joint)
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def joint_frames21(pts_mm, side):
    """[(origin, x, y, z)] in mm for 21 MediaPipe-ordered points: z runs along
    the bone toward the wrist (the wrist's toward the forearm), y points out of
    the back of the hand, x = y cross z. Unit vectors, right-handed."""
    P = np.asarray(pts_mm, float)
    back = _unit(np.cross(P[5] - P[0], P[17] - P[0])) * (-1.0 if side == "right" else 1.0)
    frames = []
    for i, parent in enumerate(MP_PARENT):
        z = _unit(P[0] - P[9]) if parent is None else _unit(P[parent] - P[i])
        y = _unit(back - z * float(np.dot(back, z)))
        x = np.cross(y, z)
        frames.append((P[i], x, y, z))
    return frames


def draw_axes(img, frames, proj):
    """Three short coloured lines per joint, through `proj` (mm -> pixel or None)."""
    for i, (o, x, y, z) in enumerate(frames):
        length = WRIST_AXIS_MM if i == 0 else AXIS_MM
        po = proj(o)
        if po is None:
            continue
        for axis, v in (("x", x), ("y", y), ("z", z)):
            pe = proj(o + v * length)
            if pe is not None:
                cv2.line(img, po, pe, AXIS_COLOURS[axis], 2, cv2.LINE_AA)


def joint_angles21(pts_mm):
    """{joint index: degrees} the bend at every joint that has a bone coming in
    and a bone going out (the knuckle, middle and end joints of each finger,
    and the three thumb joints): 0 is straight, 90 is a right angle."""
    P = np.asarray(pts_mm, float)
    child = {p: i for i, p in enumerate(MP_PARENT) if p not in (None, 0)}
    angles = {}
    for i, p in enumerate(MP_PARENT):
        if p is None or i not in child:
            continue                                             # the wrist, or a fingertip
        a = _unit(P[i] - P[p])
        b = _unit(P[child[i]] - P[i])
        angles[i] = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))
    return angles


def draw_angles(img, pts_mm, proj):
    """Each joint's bend in degrees, as a small number beside the joint."""
    P = np.asarray(pts_mm, float)
    for i, deg in joint_angles21(P).items():
        po = proj(P[i])
        if po is None:
            continue
        text = f"{deg:.0f}"
        (w, h), _ = cv2.getTextSize(text, FONT, 0.48, 1)
        x, y = po[0] + 9, po[1] - 7
        cv2.rectangle(img, (x - 2, y - h - 2), (x + w + 2, y + 2), (0, 0, 0), -1)
        put(img, text, (x, y), 0.48, WHITE, 1)


def palm_projector(pts_m, side, wrist_px, mm_per_px=MM_PER_PX):
    """The glove card's view (cam_hand.demo_view.palm_view_mm) as a function of
    any point in metres, so axes can be drawn in the same picture as the hand."""
    P = np.asarray(pts_m, float)
    origin = P[0].copy()
    B = palm_basis(P - origin)                                   # columns: up, toward pinky, normal
    flip = 1.0 if str(side).lower().startswith("l") else -1.0

    def proj(q_m):
        local = (np.asarray(q_m, float) - origin) @ B
        across, up = flip * local[1] * 1000.0, local[0] * 1000.0
        return int(round(wrist_px[0] + across / mm_per_px)), int(round(wrist_px[1] - up / mm_per_px))

    return proj


def leap_pts21(hand):
    """A sensor hand's wrist and five chains as 21 MediaPipe-ordered points in mm."""
    pts = [hand["wrist"]]
    for chain in hand["chains"]:
        pts.extend(chain[1:5])
    return pts


def axes_legend(panel, axes, angles):
    """x y z in their colours at the top right; the words only where the card is wide enough."""
    wide = panel.shape[1] >= 500
    x = panel.shape[1] - 24
    y = 38 if wide else 64                                       # a narrow card's title would collide
    if axes:
        for axis in ("z", "y", "x"):
            w = cv2.getTextSize(axis, FONT, 0.55, 2)[0][0]
            put(panel, axis, (x - w, y), 0.55, AXIS_COLOURS[axis], 2)
            x -= w + 14
        if wide:
            put_right(panel, "axes at every joint", x - 4, y, 0.45, LABEL, 1)
    if angles and wide:
        put_right(panel, "numbers: bend in degrees", panel.shape[1] - 24, 62, 0.45, LABEL, 1)


# ---------------------------------------------------------------- the picture cards
def webcam_panel(frame, hands):
    """The webcam picture, mirrored, with every detected hand drawn: WEB_W x PANEL_H."""
    if frame is None:
        panel = card(WEB_W, PANEL_H)
        put_centre(panel, "no webcam picture", WEB_W / 2, PANEL_H / 2, 0.8, LABEL, 1)
        return panel
    canvas = frame.copy()
    for h in hands:
        draw_mp_hand(canvas, h.img)
    canvas = cv2.resize(canvas, (WEB_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
    return cv2.flip(canvas, 1)                                   # a mirror: your left hand on the left


def leap_panel(img, hands, to_px, axes=True, angles=False):
    """The infrared picture, lifted and mirrored, with every tracked hand drawn: IR_W x IR_W.

    The picture is mirrored at the end, so the angle numbers are drawn on the
    mirrored copy (text must read the right way round)."""
    if img is None:
        panel = card(IR_W, PANEL_H)
        put_centre(panel, "no infrared picture", IR_W / 2, PANEL_H / 2, 0.8, LABEL, 1)
        return panel
    img = cv2.resize(img, (IR_W, IR_W), interpolation=cv2.INTER_LINEAR)
    img = cv2.convertScaleAbs(img, alpha=2.2, beta=8)            # the IR image is dark as captured
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for h in hands:
        draw_leap_hand(canvas, h, to_px)
        if axes:
            draw_axes(canvas, joint_frames21(leap_pts21(h), h["side"]), to_px)
    canvas = cv2.flip(canvas, 1)
    if angles:
        def mirrored(p):
            q = to_px(p)
            return None if q is None else (IR_W - 1 - q[0], q[1])
        for h in hands:
            draw_angles(canvas, leap_pts21(h), mirrored)
    return canvas


GLOVE_WRIST_PX = {"left": (int(GLOVE_W * 0.27), PANEL_H - 72), "right": (int(GLOVE_W * 0.73), PANEL_H - 72)}


def glove_panel(glove_hands, axes=True, angles=False):
    """Each glove's hand drawn looking at its palm, fingers up: GLOVE_W x PANEL_H.

    glove_hands: {'left': 21 points in metres (wrist at 0) or None, 'right': ...}"""
    panel = card(GLOVE_W, PANEL_H)
    for side in SIDES:
        pts = glove_hands.get(side)
        wrist = GLOVE_WRIST_PX[side]
        colour = LABEL if pts is None else WHITE
        if side == "left":                                       # labels in the top corners, clear of the hands
            put(panel, "left glove", (24, TOP_BAND + 24), 0.5, colour, 1)
        else:
            put_right(panel, "right glove", GLOVE_W - 24, TOP_BAND + 24, 0.5, colour, 1)
        if pts is not None:
            draw_palm_view(panel, pts, side, wrist, GREEN)
            proj = palm_projector(pts, side, wrist)
            pts_mm = np.asarray(pts, float) * 1000.0
            proj_mm = lambda q: proj(np.asarray(q) / 1000.0)
            if axes:
                draw_axes(panel, joint_frames21(pts_mm, side), proj_mm)
            if angles:
                draw_angles(panel, pts_mm, proj_mm)
    return panel


# ---------------------------------------------------------------- status lines
def webcam_status(hands, opened, reason=""):
    if not opened:
        return f"webcam not available: {reason}" if reason else "webcam not available", RED
    if not hands:
        return "no hand in the picture yet", AMBER
    return ("hand found, 21 points" if len(hands) == 1 else f"{len(hands)} hands found, 21 points each"), GREEN


def leap_status(hands, connected, reason=""):
    """(text, colour) for the line under the sensor panel."""
    if not connected:
        return f"sensor not connected: {reason}" if reason else "sensor not connected", RED
    if not hands:
        return "no hand over the sensor yet", AMBER
    if len(hands) > 1:
        return "both hands tracked in 3D", GREEN
    cm = hands[0]["palm"][1] / 10.0
    return f"hand tracked in 3D   {cm:.0f} cm above the sensor", GREEN


def glove_status(glove_hands, opened, reason=""):
    if not opened:
        return f"glove stream not available: {reason}" if reason else "glove stream not available", RED
    live = [s for s in SIDES if glove_hands.get(s) is not None]
    if not live:
        return "no glove data yet: start XR Train and connect a glove", AMBER
    if len(live) == 2:
        return "both gloves streaming", GREEN
    return f"{live[0]} glove streaming", GREEN


# ---------------------------------------------------------------- the numbers
def _cm(p, scale):
    return tuple(round(v * scale, 1) + 0.0 for v in p)            # + 0.0 turns -0.0 into 0.0


def sensor_points(hand):
    """{name: (x, y, z) cm from the sensor} for a tracked hand: palm and five tips."""
    out = {"palm": _cm(hand["palm"], 0.1)}
    for digit, name in enumerate(POINT_NAMES[1:]):
        out[name] = _cm(hand["chains"][digit][-1], 0.1)
    return out


def glove_points(pts21):
    """{name: (x, y, z) cm from the wrist} for a glove hand's 21 points in
    metres: the palm is the middle of the four knuckles."""
    P = np.asarray(pts21, float)
    out = {"palm": _cm(P[list(MP_KNUCKLES)].mean(axis=0), 100.0)}
    for name, idx in MP_TIPS.items():
        out[name] = _cm(P[idx], 100.0)
    return out


def hands_by_side(hands):
    """{'left': hand or None, 'right': hand or None}: the first hand the sensor
    labels with each side. The label is the tracker's; with one hand in view it
    is sometimes mirrored, with both hands up it sorts itself out."""
    out = {"left": None, "right": None}
    for h in hands:
        if out.get(h["side"]) is None:
            out[h["side"]] = h
    return out


def joint_bends(pts_mm):
    """{'knuckle' | 'middle' | 'end': [thumb, index, middle, ring, pinky]} the
    bend in degrees at that joint of each finger (the thumb's three joints
    fall in the same three rows)."""
    angles = joint_angles21(pts_mm)
    return {row: [angles[i] for i in joints] for row, joints in BEND_ROWS.items()}


def table_rows(points_by_side, bends_by_side=None):
    """What a table prints, top to bottom, for the hands it has: ('hand', side),
    then ('x' | 'y' | 'z', [six texts]) and, when bends are given, ('knuckle' |
    'middle' | 'end', ['-', five texts]) for each present side, left first."""
    rows = []
    for side in SIDES:
        pts = points_by_side.get(side)
        if not pts:
            continue
        rows.append(("hand", side))
        for axis in range(3):
            rows.append(("xyz"[axis], [f"{pts[name][axis]:.1f}" for name in POINT_NAMES]))
        bends = (bends_by_side or {}).get(side)
        if bends:
            for name in BEND_ROWS:
                rows.append((name, ["-"] + [f"{deg:.0f}" for deg in bends[name]]))
    return rows


# the table's geometry: a label column, then six point columns
_UNIT_X, _AXIS_X = 24, 70
_COL_RIGHT = [290 + 126 * k for k in range(6)]        # right edge of each point column
_ROWS_Y0 = 140
_UNITS = {"x": "cm", "knuckle": "degrees"}            # printed once, on the first row of each group


def numbers_table(points_by_side, bends_by_side, title, caption, empty_text):
    """One table card, TABLE_W x TABLE_H: rows only for the hands it has.

    points_by_side: {'left': {name: (x, y, z)} or None, 'right': ...}
    bends_by_side:  {'left': {'knuckle': [5 degrees], ...} or None, 'right': ...}"""
    panel = card(TABLE_W, TABLE_H)
    put(panel, title, (24, 40), 0.8, WHITE, 2)
    put(panel, caption, (24, 68), 0.5, LABEL, 1)
    rows = table_rows(points_by_side, bends_by_side)
    if not rows:
        put_centre(panel, empty_text, TABLE_W / 2, TABLE_H / 2 + 20, 0.7, LABEL, 1)
        return panel
    for word, right in zip(COLUMN_WORDS, _COL_RIGHT):
        put_right(panel, word, right, 106, 0.5, LABEL, 1)
    pitch = min(38, int((TABLE_H - _ROWS_Y0 - 10) / len(rows)))
    y = _ROWS_Y0
    for kind, payload in rows:
        if kind == "hand":
            put(panel, f"{payload.capitalize()} hand", (24, y), 0.6, WHITE, 2)
        else:
            if kind in _UNITS:
                put(panel, _UNITS[kind], (_UNIT_X, y), 0.42, LABEL, 1)
            put(panel, kind, (_AXIS_X, y), 0.5, LABEL, 1)
            for text, right in zip(payload, _COL_RIGHT):
                put_right(panel, text, right, y, 0.66, VALUE if text != "-" else LABEL, 1)
        y += pitch
    return panel


def sensor_table(leap_hands):
    sides = hands_by_side(leap_hands)
    pts = {s: (sensor_points(h) if h is not None else None) for s, h in sides.items()}
    bends = {s: (joint_bends(leap_pts21(h)) if h is not None else None) for s, h in sides.items()}
    return numbers_table(pts, bends, "3D sensor",
                         "position in centimetres from the sensor (x sideways, y height, z toward you); "
                         "bend of each finger joint in degrees, 0 = straight",
                         "no hand over the sensor yet")


def glove_table(glove_hands):
    pts = {s: (glove_points(p) if p is not None else None) for s, p in glove_hands.items()}
    bends = {s: (joint_bends(np.asarray(p, float) * 1000.0) if p is not None else None) for s, p in glove_hands.items()}
    return numbers_table(pts, bends, "Glove",
                         "position in centimetres from the wrist (the glove knows the hand's shape, not its place); "
                         "bend in degrees, 0 = straight",
                         "no glove data yet")


# ---------------------------------------------------------------- the window
def compose(web_frame, web_hands, ir_img, leap_hands, to_px, glove_hands, web_state, leap_state, glove_state,
            axes=True, angles=False):
    """The whole window: header, three picture cards, two table cards, footer.

    web_state / leap_state / glove_state: (opened: bool, reason: str).
    axes: draw the x, y, z frame at every joint of the sensor and glove hands.
    angles: also print each joint's bend beside it on the picture (the tables
    always list the bends)."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
    put(canvas, "Hand tracking", (MARGIN + 8, 44), 1.2, WHITE, 2)
    put(canvas, "three ways a computer can know what a hand is doing.  Nothing is recorded.", (MARGIN + 8, 74), 0.55, LABEL, 1)

    cards = (
        (WEB_X, WEB_W, webcam_panel(web_frame, web_hands), web_frame is not None,
         "Webcam", "an ordinary camera: the software finds the hand and marks 21 points",
         webcam_status(web_hands, *web_state)),
        (IR_X, IR_W, leap_panel(ir_img, leap_hands, to_px, axes, angles), ir_img is not None,
         "3D infrared sensor", "two infrared eyes, the hand in 3D",
         leap_status(leap_hands, *leap_state)),
        (GLOVE_X, GLOVE_W, glove_panel(glove_hands, axes, angles), False,
         "Glove", "stretch sensors in the fabric feel how far each finger bends",
         glove_status(glove_hands, *glove_state)),
    )
    for x, w, panel, on_picture, title, caption, (text, colour) in cards:
        title_strip(panel, title, caption, on_picture)
        status_strip(panel, text, colour, on_picture)
        if x != WEB_X:
            axes_legend(panel, axes, angles)
        canvas[PANEL_Y:PANEL_Y + PANEL_H, x:x + w] = panel

    canvas[TABLE_Y:TABLE_Y + TABLE_H, SENSOR_TABLE_X:SENSOR_TABLE_X + TABLE_W] = sensor_table(leap_hands)
    canvas[TABLE_Y:TABLE_Y + TABLE_H, GLOVE_TABLE_X:GLOVE_TABLE_X + TABLE_W] = glove_table(glove_hands)

    put_right(canvas, "q closes   f full screen   a axes on/off   n bend numbers on the picture on/off",
              CANVAS_W - MARGIN - 8, CANVAS_H - 8, 0.45, LABEL, 1)
    return canvas


# ---------------------------------------------------------------- sources
class Webcam:
    """The laptop's webcam through OpenCV (cam_hand.capture picks the backend)."""

    def __init__(self, index="auto"):
        from cam_hand.capture import open_camera
        indices = [int(index)] if index != "auto" else [0, 1, 2, 3]
        self.cap = None
        self.index = None
        errors = []
        for i in indices:
            try:
                self.cap = open_camera(i)
                self.index = i
                break
            except RuntimeError as ex:
                errors.append(str(ex).splitlines()[0])
        if self.cap is None:
            raise RuntimeError(errors[-1] if errors else "no webcam found")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class Detector:
    """MediaPipe's hand landmarker in video mode, with strictly increasing stamps."""

    def __init__(self):
        from cam_hand.landmarks import HandTracker
        self.tracker = HandTracker(num_hands=2)
        self._last_ms = 0

    def detect(self, frame):
        ms = int(time.monotonic() * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        try:
            return self.tracker.detect(frame, ms)
        except Exception as ex:                                  # one bad frame must not end the demo
            print("hand model skipped a frame:", ex)
            return []

    def close(self):
        self.tracker.close()


class LeapSource:
    """Read-only client of the Ultraleap tracking service: the latest infrared
    picture and the latest tracked hands, filled on the SDK's own thread."""

    def __init__(self):
        import leap
        from leap import enums
        from leapc_cffi import ffi, libleapc

        self._ffi = ffi
        self._lib = libleapc
        self.lock = threading.Lock()
        self.img = None
        self.img_t = None
        self.hands = []
        self.hands_t = 0.0
        self.fps = 0.0
        source = self

        class Listener(leap.Listener):
            def on_image_event(self, e):
                try:
                    img = source._image_to_numpy(e.image[0])
                except Exception:
                    return
                with source.lock:
                    source.img = img
                    source.img_t = time.time()

            def on_tracking_event(self, e):
                hands = []
                for h in e.hands:
                    chains = []
                    for d in h.digits:
                        b = d.bones
                        chains.append([_vec(b[0].prev_joint), _vec(b[1].prev_joint), _vec(b[2].prev_joint),
                                       _vec(b[3].prev_joint), _vec(b[3].next_joint)])
                    hands.append(dict(side="left" if h.type == enums.HandType.Left else "right",
                                      palm=_vec(h.palm.position), wrist=_vec(h.arm.next_joint),
                                      elbow=_vec(h.arm.prev_joint), chains=chains))
                with source.lock:
                    source.hands = hands
                    source.hands_t = time.time()
                    source.fps = float(e.framerate or 0.0)

        self.conn = leap.Connection(listeners=[Listener()])
        self.conn.connect()
        self.conn.set_tracking_mode(enums.TrackingMode.Desktop)
        try:
            self.conn.set_policy_flags(flags_to_set=[enums.PolicyFlag.Images])
        except Exception as ex:
            print("could not enable the infrared picture:", ex)
        self.ptr = self.conn.get_connection_ptr()
        self.device = self._device_name()

    def _device_name(self):
        try:
            devs = self.conn.get_devices()
        except Exception:
            return "device not listed"
        if not devs:
            return ""
        try:
            return str(devs[0].get_info().product_id or "Ultraleap")
        except Exception:
            return "Ultraleap"

    def _image_to_numpy(self, image):
        c = image.c_data
        p = c.properties
        n = p.width * p.height * p.bpp
        ptr = self._ffi.cast("uint8_t *", c.data) + c.offset
        a = np.frombuffer(self._ffi.buffer(ptr, n), dtype=np.uint8).copy()   # copy now: LeapC reuses the buffer
        return a.reshape(p.height, p.width) if p.bpp == 1 else a.reshape(p.height, p.width, p.bpp)[:, :, 0]

    def to_px(self, p):
        """Tracking-space millimetres -> pixel in the (unmirrored, upscaled) left image."""
        x, y, z = p
        if y <= 5.0:
            return None
        v = self._ffi.new("LEAP_VECTOR*")
        v.x = -(x - BASELINE_HALF_MM) / y
        v.y = z / y
        v.z = 1.0
        r = self._lib.LeapRectilinearToPixel(self.ptr, LEFT_CAMERA, v[0])
        if not (math.isfinite(r.x) and math.isfinite(r.y)):
            return None
        return int(r.x * IR_SCALE), int(r.y * IR_SCALE)

    def snapshot(self):
        """(infrared image or None, fresh hands, framerate)."""
        with self.lock:
            img = None if self.img is None else self.img.copy()
            hands = list(self.hands) if time.time() - self.hands_t < FRESH_S else []
            return img, hands, self.fps

    def close(self):
        try:
            self.conn.disconnect()
        except Exception:
            pass


def _vec(v):
    return (v.x, v.y, v.z)


class GloveSource:
    """XR Train's OSC stream, read the way every glove recorder here reads it:
    xr_hand.receiver (UDP port 9002) -> parser -> 21 wrist-centred points."""

    def __init__(self, port=9002, mock=False):
        from xr_hand.keypoints21 import frame_to_keypoints21
        from xr_hand.parser import parse_hand_message
        self._to21 = frame_to_keypoints21
        self._parse = parse_hand_message
        if mock:
            from leap_hand.live import MockGlove
            self.receivers = [MockGlove("left"), MockGlove("right")]
        else:
            from xr_hand.receiver import OSCHandReceiver
            self.receivers = [OSCHandReceiver(port=port)]
        for r in self.receivers:
            r.start()
        self.latest = {"left": None, "right": None}
        self.at = {"left": 0.0, "right": 0.0}
        self.packets = {"left": 0, "right": 0}

    def poll(self):
        """Drain every packet that arrived since the last call; keep the newest per hand."""
        for r in self.receivers:
            for item in r.drain(256):
                hand, raw = item
                try:
                    frame = self._parse(raw, hand_side_hint=hand)
                    pts = self._to21(frame)
                except Exception:
                    continue                                     # a malformed packet is not data
                side = frame.hand_side
                if side not in self.latest:
                    continue
                self.latest[side] = pts
                self.at[side] = float(getattr(item, "recv_time", 0.0) or time.time())
                self.packets[side] += 1

    def hands(self):
        """{'left': 21 points or None, 'right': ...}: only streams seen in the last FRESH_S."""
        now = time.time()
        return {s: (p if p is not None and now - self.at[s] < FRESH_S else None) for s, p in self.latest.items()}

    def close(self):
        for r in self.receivers:
            try:
                r.stop()
            except Exception:
                pass


# ---------------------------------------------------------------- synthetic sources (no devices)
class _Hand:
    def __init__(self, img):
        self.img = img
        self.hand_side = "left"
        self.score = 1.0
        self.world = []


def mock_webcam_frame(t=0.0):
    """A 640 x 480 picture with a drawn open hand, so the drawing path runs without a camera."""
    frame = np.full((480, 640, 3), (70, 60, 50), np.uint8)
    cv2.putText(frame, "synthetic webcam", (16, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1, cv2.LINE_AA)
    return frame


def mock_mp_hand(t=0.0):
    """21 image-space points of an open hand, fingers up, wrist at the bottom; wiggles with t."""
    cx, cy = 320 + 20 * math.sin(t), 300
    pts = [[cx, cy + 120, 0.0]]
    for ang, length in [(-70, 110), (-20, 160), (0, 170), (18, 160), (38, 140)]:
        rad = math.radians(ang)
        for j in range(1, 5):
            r = length * j / 4
            pts.append([cx + r * math.sin(rad), cy + 60 - r * math.cos(rad), 0.0])
    return _Hand(pts)


def mock_leap_hand(t=0.0, side="left", dx=-20.0):
    """One tracked hand in tracking-space mm: palm 220 mm up, fingers away from
    the user, the thumb toward -x for a left hand and +x for a right hand."""
    palm = (dx + 20.0 * math.sin(t), 220.0, 0.0)
    wrist = (palm[0], 215.0, 70.0)
    elbow = (palm[0], 200.0, 300.0)
    chains = []
    mirror = -1.0 if side == "right" else 1.0
    for ang, length in [(-60, 80), (-15, 95), (0, 100), (12, 95), (28, 80)]:
        rad = math.radians(mirror * ang)
        chains.append([(wrist[0] + length * j / 4 * math.sin(rad), 222.0, wrist[2] - 20 - length * j / 4 * math.cos(rad))
                       for j in range(1, 6)])
    return dict(side=side, palm=palm, wrist=wrist, elbow=elbow, chains=chains)


def mock_leap_hands(t=0.0):
    """Both synthetic hands, a left one and a right one side by side."""
    return [mock_leap_hand(t, "left", -20.0), mock_leap_hand(t, "right", 80.0)]


def mock_to_px(p):
    """A pinhole for the synthetic sensor: x and z in mm onto the 384 px image, scaled like the real one."""
    x, y, z = p
    if y <= 5.0:
        return None
    f = 300.0
    return int((SENSOR_PX / 2 - f * (x - BASELINE_HALF_MM) / y) * IR_SCALE), int((SENSOR_PX / 2 + f * z / y) * IR_SCALE)


def mock_ir_image(t=0.0):
    rng = np.random.default_rng(int(t * 10) % 1000)
    img = (rng.random((SENSOR_PX, SENSOR_PX)) * 30 + 10).astype(np.uint8)
    cv2.circle(img, (SENSOR_PX // 2, SENSOR_PX // 2), 60, 90, -1)
    return img


def mock_glove_pts(side="right", t=0.0):
    """21 wrist-centred points in metres of an open glove hand, fingers up (+y),
    thumb toward +x on a right hand and -x on a left one, curling a little with t."""
    pts = [(0.0, 0.0, 0.0)]
    mirror = 1.0 if side == "right" else -1.0
    curl = 0.15 * (1 + math.sin(t))
    for ang, length in [(-55, 0.11), (-15, 0.17), (0, 0.18), (12, 0.17), (28, 0.15)]:
        rad = math.radians(mirror * ang)
        for j in range(1, 5):
            r = length * j / 4
            pts.append((r * math.sin(rad), r * math.cos(rad) * (1 - curl * j / 4), -0.02 * curl * j))
    return pts


def mock_glove_hands(t=0.0):
    return {"left": mock_glove_pts("left", t), "right": mock_glove_pts("right", t)}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--webcam", default="auto", help="OpenCV camera index, or auto (default): the first that opens")
    ap.add_argument("--port", type=int, default=9002, help="UDP port XR Train streams the gloves to (default 9002)")
    ap.add_argument("--no-webcam", action="store_true", help="leave the webcam panel off")
    ap.add_argument("--no-leap", action="store_true", help="leave the sensor panel off")
    ap.add_argument("--no-glove", action="store_true", help="leave the glove panel off")
    ap.add_argument("--mock", action="store_true", help="synthetic pictures and hands, no devices (testing)")
    ap.add_argument("--fullscreen", action="store_true", help="open full screen (f toggles)")
    ap.add_argument("--seconds", type=float, default=0.0, help="close by itself after this long (0 = never)")
    ap.add_argument("--frames", type=int, default=0, help="stop after drawing this many frames (0 = never)")
    ap.add_argument("--no-window", action="store_true", help="draw without a window (with --snapshot)")
    ap.add_argument("--snapshot", default=None, help="write the last frame drawn as a PNG")
    args = ap.parse_args(argv)

    webcam = detector = leap_src = glove_src = None
    web_state = (False, "switched off")
    leap_state = (False, "switched off")
    glove_state = (False, "switched off")
    if args.mock:
        web_state = leap_state = glove_state = (True, "")
        print("webcam: synthetic")
        print("3D sensor: synthetic")
        print("glove: synthetic")
    else:
        if not args.no_webcam:
            try:
                webcam = Webcam(args.webcam)
                detector = Detector()
                web_state = (True, "")
                print(f"webcam: open (index {webcam.index}, {webcam.size[0]}x{webcam.size[1]}); hand model loaded")
            except Exception as ex:
                web_state = (False, str(ex).splitlines()[0][:70])
                print("webcam: not available:", ex)
        if not args.no_leap:
            try:
                leap_src = LeapSource()
                if leap_src.device == "":
                    leap_state = (False, "no sensor found on USB")
                    print("3D sensor: service reached, but no sensor found on USB")
                else:
                    leap_state = (True, "")
                    print(f"3D sensor: connected ({leap_src.device})")
            except Exception as ex:
                leap_state = (False, "tracking service not running")
                print("3D sensor: not connected:", ex)
        if not args.no_glove:
            try:
                glove_src = GloveSource(port=args.port)
                glove_state = (True, "")
                print(f"glove: listening on port {args.port} for XR Train")
            except Exception as ex:
                glove_state = (False, f"port {args.port} busy, close other glove programs")
                print("glove: stream not available:", ex)

    if not args.no_window:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1500, 844)
        if args.fullscreen:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = args.fullscreen
    axes = True
    angles = False                 # the tables carry the bends; n puts them on the picture too
    t0 = time.time()
    frames = 0
    last = None
    try:
        while True:
            t = time.time() - t0
            web_frame, web_hands = None, []
            ir_img, leap_hands = None, []
            glove_hands = {"left": None, "right": None}
            to_px = mock_to_px
            if args.mock:
                web_frame, web_hands = mock_webcam_frame(t), [mock_mp_hand(t)]
                ir_img, leap_hands = mock_ir_image(t), mock_leap_hands(t)
                glove_hands = mock_glove_hands(t)
            else:
                if webcam is not None:
                    web_frame = webcam.read()
                    if web_frame is not None and detector is not None:
                        web_hands = detector.detect(web_frame)
                if leap_src is not None:
                    ir_img, leap_hands, _fps = leap_src.snapshot()
                    to_px = leap_src.to_px
                if glove_src is not None:
                    glove_src.poll()
                    glove_hands = glove_src.hands()
            last = compose(web_frame, web_hands, ir_img, leap_hands, to_px, glove_hands,
                           web_state, leap_state, glove_state, axes, angles)
            frames += 1
            if not args.no_window:
                cv2.imshow(WIN, last)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("a"):
                    axes = not axes
                if k == ord("n"):
                    angles = not angles
                if k == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            elif webcam is None:
                time.sleep(0.01)
            if args.frames and frames >= args.frames:
                break
            if args.seconds and time.time() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Drew {frames} frame(s).")
        if args.snapshot and last is not None:
            Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.snapshot, last)
            print(f"Snapshot written: {args.snapshot}")
        for src in (detector, webcam, leap_src, glove_src):
            if src is not None:
                src.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
