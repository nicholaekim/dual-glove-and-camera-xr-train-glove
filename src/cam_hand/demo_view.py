"""Drawing for the demo window: the glove, camera and fused hands side by side.

`scripts/demo.py` feeds this module one `FusedFrame` at a time (from a
replayed session or from the live sensors) and asks it for a picture. Nothing
here fuses anything: the three hands are the two inputs of one
`LiveFusion.step` (`FusedFrame.glove_in` and `FusedFrame.cam_in`) and its
output (`FusedFrame.pts`).

How a hand is drawn

  Every hand is 21 wrist-centred points in metres, MediaPipe-21 order. Each
  one is turned into its OWN palm frame (`fusion.palm_basis`: up is wrist to
  middle knuckle, across is index knuckle to pinky knuckle) and drawn as an
  orthographic view looking at the palm: fingers up, the thumb on the
  viewer's right for a right hand and on the left for a left hand. So wrist
  rotation is taken out and the three hands can be compared finger by
  finger. The scale is fixed (`MM_PER_PX` millimetres per pixel) and the
  same for all three, so a longer bone on screen is a longer bone.

  A curled finger folds towards the viewer, so in this view it looks
  shorter, with its tip nearer the palm.

What the badges say

  Under the fused hand, one line per finger says which sensor supplied that
  finger in this frame: its curl (the glove, unless the rail override gave
  it to the camera) and its spread (the camera when its gates passed, else
  the glove), and for the thumb, its direction. Then which fingers the rail
  override is holding, whether the camera is fresh, and the glove's rate.

The pose guess is a nearest-centroid classifier on `features.all_features`
of the fused hand, the report's own features and classifier
(`fuse_poses.loo_table`), with centroids from the per-take means of a fused
session (`CentroidClassifier`).
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from cam_hand.features import ALL_COLS, all_features
from cam_hand.fusion import (
    FINGER_CHAINS,
    SPREAD_FINGERS,
    SRC_RAIL,
    palm_basis,
)
from cam_hand.live_fusion import STALE_S

# Colours, BGR. The three hands.
GLOVE_BGR = (0x34, 0x68, 0xEB)      # #eb6834 orange
CAMERA_BGR = (0xD6, 0x78, 0x2A)     # #2a78d6 blue
FUSED_BGR = (0x7A, 0xAF, 0x1B)      # #1baf7a green
GREY_BGR = (105, 105, 105)
TEXT_BGR = (235, 235, 235)
DIM_BGR = (160, 160, 160)
BG_BGR = (26, 24, 22)
PANEL_BGR = (40, 37, 34)
GOOD_BGR = (90, 210, 90)
BAD_BGR = (70, 70, 235)
STREAM_BGR = {"glove": GLOVE_BGR, "camera": CAMERA_BGR, "fused": FUSED_BGR}

# Fixed scale, the same for every hand. The tallest hand in sync_day2
# reaches 189 mm above the wrist in this view (236 px of the 240 above the
# wrist), so every hand of that session fits its panel.
MM_PER_PX = 0.8
# A stream not seen for this long is drawn grey, with "no hand".
NO_HAND_S = 0.5
# Fused frames averaged for the pose guess (about a quarter second at 60 Hz),
# so the guess does not flicker from frame to frame.
FEATURE_WINDOW = 15

FONT = cv2.FONT_HERSHEY_SIMPLEX

# MediaPipe's own hand drawing: each finger chain from the wrist, plus the
# knuckle line.
BONES: List[Tuple[int, int]] = (
    [(0, FINGER_CHAINS["thumb"][0]), (0, FINGER_CHAINS["index"][0]),
     (0, FINGER_CHAINS["pinky"][0])]
    + [(chain[i], chain[i + 1]) for chain in FINGER_CHAINS.values()
       for i in range(len(chain) - 1)]
    + [(5, 9), (9, 13), (13, 17)])
TIPS = [chain[-1] for chain in FINGER_CHAINS.values()]


# --- the projection ----------------------------------------------------------

def palm_view_mm(pts, hand_side: str) -> np.ndarray:
    """21 x 2 millimetres: the hand seen looking at its palm, wrist at 0.

    Column 0 is across (positive to the viewer's right), column 1 is up
    (wrist to middle knuckle). The hand is expressed in its own palm basis,
    so the view does not depend on how the wrist was turned. The across axis
    of `palm_basis` points from the index side to the pinky side on both
    hands; looking at a right palm the pinky is on the viewer's left, so a
    right hand is flipped and a left hand is not.
    """
    P = np.asarray(pts, float)
    P = P - P[0]
    local = P @ palm_basis(P)           # columns: up, towards pinky, normal
    across = local[:, 1]
    if not str(hand_side).lower().startswith("l"):
        across = -across
    return np.column_stack([across, local[:, 0]]) * 1000.0


def to_pixels(view_mm: np.ndarray, wrist_px: Tuple[float, float],
              mm_per_px: float = MM_PER_PX) -> np.ndarray:
    """Millimetres (x right, y up) -> integer pixels (y down), wrist at
    `wrist_px`."""
    v = np.asarray(view_mm, float)
    x = wrist_px[0] + v[:, 0] / mm_per_px
    y = wrist_px[1] - v[:, 1] / mm_per_px
    return np.rint(np.column_stack([x, y])).astype(int)


def draw_hand(img: np.ndarray, pts, hand_side: str,
              wrist_px: Tuple[float, float], colour,
              mm_per_px: float = MM_PER_PX) -> np.ndarray:
    """Bones as lines, joints as dots, tips a little larger. Returns the
    pixel coordinates drawn."""
    px = to_pixels(palm_view_mm(pts, hand_side), wrist_px, mm_per_px)
    for a, b in BONES:
        cv2.line(img, tuple(px[a]), tuple(px[b]), colour, 3, cv2.LINE_AA)
    for i, p in enumerate(px):
        r = 5 if i in TIPS else 3
        cv2.circle(img, tuple(p), r + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), r, (240, 240, 240) if i == 0 else colour,
                   -1, cv2.LINE_AA)
    return px


# --- text --------------------------------------------------------------------

def put(img, text, org, scale=0.5, colour=TEXT_BGR, thick=1) -> int:
    """Text, as `put` in scripts/leap/camera_view.py but without its dark
    outline: the background here is already dark, and OpenCV 5 draws a
    thicker stroke as a wider font, so an outline pokes out past the text.
    Returns the x just after the text."""
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)
    return org[0] + cv2.getTextSize(text, FONT, scale, thick)[0][0]


def put_segments(img, segments: Sequence[Tuple[str, tuple]], org,
                 scale=0.5, thick=1) -> int:
    """Several pieces of text on one line, each in its own colour."""
    x, y = org
    for text, colour in segments:
        x = put(img, text, (x, y), scale, colour, thick)
    return x


def text_width(text: str, scale=0.5, thick=1) -> int:
    return cv2.getTextSize(text, FONT, scale, thick)[0][0]


# --- the badges --------------------------------------------------------------

def source_word(source: Optional[str]) -> str:
    """A dof_source value in the badge's words."""
    if source is None:
        return "glove"
    if source == SRC_RAIL:
        return "camera (override)"
    return "camera" if str(source).startswith("camera") else "glove"


def _word_colour(word: str):
    return CAMERA_BGR if word.startswith("camera") else GLOVE_BGR


def badge_cells(dof_source: Mapping[str, str],
                rail_active: Sequence[str] = ()
                ) -> List[List[List[Tuple[str, tuple]]]]:
    """The badge lines under the fused hand: per line, its cells (drawn in
    fixed columns, `BADGE_COLS`), each cell a list of coloured pieces.

    One line for the thumb (its direction), one per finger (curl and
    spread), then the fingers the rail override is holding. The keys read
    are `fusion.GATED_DOFS`: "thumb", "spread <finger>", "curl <finger>".
    """
    def said(label, word):
        return [(label, DIM_BGR), (word, _word_colour(word))]

    lines = [[[("thumb", TEXT_BGR)],
              said("direction: ", source_word(dof_source.get("thumb")))]]
    for finger in SPREAD_FINGERS:
        lines.append([
            [(finger, TEXT_BGR)],
            said("curl: ", source_word(dof_source.get(f"curl {finger}"))),
            said("spread: ",
                 source_word(dof_source.get(f"spread {finger}")))])
    held = ", ".join(rail_active) if rail_active else "none"
    lines.append([[("override: ", DIM_BGR),
                   (held, CAMERA_BGR if rail_active else TEXT_BGR)]])
    return lines


def badge_text(dof_source: Mapping[str, str],
               rail_active: Sequence[str] = ()) -> List[str]:
    """`badge_cells` as plain strings, e.g.
    "index   curl: glove   spread: camera"."""
    return ["   ".join("".join(t for t, _c in cell) for cell in line)
            for line in badge_cells(dof_source, rail_active)]


def status_text(camera_fresh: bool, glove_hz: Optional[float]) -> str:
    hz = "--" if glove_hz is None else f"{glove_hz:.0f}"
    return (f"camera {'fresh' if camera_fresh else 'STALE'}   "
            f"glove {hz} Hz")


# --- the pose guess ----------------------------------------------------------

@dataclass
class Guess:
    pose: str
    distance: float
    runner_up: Optional[str]
    margin: Optional[float]       # runner-up distance minus this one


class CentroidClassifier:
    """Nearest centroid over per-take mean feature vectors.

    `samples` are `(pose, hand, take, features)`, the shape of
    `fuse_poses.FusionRun.fused_samples`, and a pose's centroid is the mean
    of its samples over both hands, as `features.loo_take_nearest_centroid`
    pools them. `exclude_take` leaves one take out, so a replayed take is
    never guessed by a centroid it helped build.
    """

    def __init__(self,
                 samples: Sequence[Tuple[str, str, str, Sequence[float]]],
                 cols: Sequence[int] = ALL_COLS, source: str = ""):
        self.samples = [(str(p), str(h), str(t),
                         np.asarray([f[i] for i in cols], float))
                        for p, h, t, f in samples]
        self.cols = list(cols)
        self.source = source
        self._cache: Dict[Optional[str], Dict[str, np.ndarray]] = {}

    @property
    def poses(self) -> List[str]:
        return sorted({p for p, _h, _t, _f in self.samples})

    def centroids(self, exclude_take: Optional[str] = None
                  ) -> Dict[str, np.ndarray]:
        if exclude_take not in self._cache:
            by_pose: Dict[str, list] = {}
            for pose, _hand, take, feats in self.samples:
                if take != exclude_take:
                    by_pose.setdefault(pose, []).append(feats)
            self._cache[exclude_take] = {p: np.mean(v, axis=0)
                                         for p, v in by_pose.items()}
        return self._cache[exclude_take]

    def guess(self, features: Optional[Sequence[float]],
              exclude_take: Optional[str] = None) -> Optional[Guess]:
        """The nearest pose, its Euclidean distance, and the margin to the
        next nearest. None with no features or no centroid."""
        if features is None:
            return None
        cents = self.centroids(exclude_take)
        if not cents:
            return None
        x = np.asarray([features[i] for i in self.cols], float)
        ranked = sorted((float(np.linalg.norm(x - c)), p)
                        for p, c in cents.items())
        d, pose = ranked[0]
        if len(ranked) > 1:
            return Guess(pose, d, ranked[1][1], ranked[1][0] - d)
        return Guess(pose, d, None, None)


def pose_features(pts, hand_side: str) -> List[float]:
    """The classifier's features for one hand: the report's `all_features`."""
    return all_features(np.asarray(pts, float), hand_side=hand_side)


# --- what one hand's row shows -----------------------------------------------

class SideState:
    """The latest glove, camera and fused hand of one side, and when each
    was last seen, on whatever clock the caller passes (the take's stamps in
    a replay, the wall clock live)."""

    def __init__(self, side: str, window: int = FEATURE_WINDOW):
        self.side = side
        self.glove: Optional[np.ndarray] = None
        self.camera: Optional[np.ndarray] = None
        self.fused: Optional[np.ndarray] = None
        self.t_glove: Optional[float] = None
        self.t_camera: Optional[float] = None
        self.frame = None
        self._stamps: deque = deque()
        self._features: deque = deque(maxlen=window)

    def update(self, frame, now: Optional[float] = None) -> None:
        now = float(frame.t_glove if now is None else now)
        self.frame = frame
        self.fused = np.asarray(frame.pts, float)
        glove = getattr(frame, "glove_in", None)
        self.glove = self.fused if glove is None else np.asarray(glove, float)
        self.t_glove = now
        cam = getattr(frame, "cam_in", None)
        if cam is not None:
            self.camera = np.asarray(cam, float)
            self.t_camera = now
        self._stamps.append(now)
        while self._stamps and self._stamps[0] <= now - 1.0:
            self._stamps.popleft()
        self._features.append(pose_features(self.fused, self.side))

    @staticmethod
    def _within(t: Optional[float], now: float, limit: float) -> bool:
        return t is not None and now - t <= limit

    def seen(self, stream: str, now: float) -> bool:
        """Was `stream` ("glove", "camera" or "fused") seen within
        `NO_HAND_S`?"""
        t = self.t_camera if stream == "camera" else self.t_glove
        return self._within(t, now, NO_HAND_S)

    def camera_fresh(self, now: float) -> bool:
        """Did the fusion find a camera partner within `STALE_S`?"""
        return self._within(self.t_camera, now, STALE_S)

    def glove_hz(self, now: float) -> Optional[float]:
        """Glove frames in the last second."""
        if self.t_glove is None:
            return None
        return float(sum(1 for t in self._stamps if now - 1.0 < t <= now))

    def mean_features(self) -> Optional[List[float]]:
        if not self._features:
            return None
        return np.mean(np.asarray(self._features, float), axis=0).tolist()


# --- the layout --------------------------------------------------------------

MARGIN = 16
GAP = 10
COL_W = (330, 330, 420)          # glove, camera, fused (+ badges)
HEADER_H = 70
PANEL_TITLE_H = 26
HAND_H = 270
WRIST_FROM_BOTTOM = 30
LINE_H = 20
INFO_H = 7 * LINE_H + 14
# x of the badge columns: finger, curl (or the thumb's direction), spread
BADGE_COLS = (0, 70, 250)
ROW_H = PANEL_TITLE_H + HAND_H + INFO_H + 12
FOOTER_H = 30
WIDTH = 2 * MARGIN + sum(COL_W) + 2 * GAP
STREAMS = ("glove", "camera", "fused")
TITLES = {"glove": "GLOVE", "camera": "CAMERA (Ultraleap)", "fused": "FUSED"}


def canvas_size(n_rows: int) -> Tuple[int, int]:
    """(width, height) of the picture for `n_rows` hands."""
    return WIDTH, HEADER_H + max(1, n_rows) * ROW_H + FOOTER_H


def even_size(size: Tuple[int, int]) -> Tuple[int, int]:
    """(width, height) rounded up to even numbers, as video codecs want."""
    w, h = (int(v) for v in size)
    return w + w % 2, h + h % 2


def pad_to(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """`img` in the top left of a picture of `size` (width, height), the rest
    the background colour. A picture already that size comes back as is; a
    larger one is cut to it."""
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    out = np.full((h, w, 3), BG_BGR, np.uint8)
    ih, iw = min(h, img.shape[0]), min(w, img.shape[1])
    out[:ih, :iw] = img[:ih, :iw]
    return out


def _col_x(k: int) -> int:
    return MARGIN + sum(COL_W[:k]) + GAP * k


def draw_tick(img, x: int, y: int, ok: bool, size: int = 14) -> None:
    """A tick (right) or a cross (wrong), drawn, because the Hershey fonts
    have neither."""
    if ok:
        pts = np.array([[x, y - size // 2], [x + size // 3, y],
                        [x + size, y - size]], np.int32)
        cv2.polylines(img, [pts], False, GOOD_BGR, 3, cv2.LINE_AA)
    else:
        cv2.line(img, (x, y - size), (x + size, y), BAD_BGR, 3, cv2.LINE_AA)
        cv2.line(img, (x, y), (x + size, y - size), BAD_BGR, 3, cv2.LINE_AA)


def _panel(img, state: SideState, stream: str, x: int, y: int, w: int,
           now: float, mm_per_px: float, prefix: str = "") -> None:
    colour = STREAM_BGR[stream]
    seen = state.seen(stream, now)
    tx = x + 2
    if prefix:
        tx = put(img, prefix, (tx, y + 19), 0.65, TEXT_BGR, 2) + 14
    put(img, TITLES[stream], (tx, y + 19), 0.6, colour if seen else GREY_BGR,
        2)
    top = y + PANEL_TITLE_H
    cv2.rectangle(img, (x, top), (x + w - 1, top + HAND_H - 1), PANEL_BGR, -1)
    # Draw on the panel's own pixels, so a hand that reaches past the panel
    # is clipped there instead of running into its neighbour.
    roi = img[top:top + HAND_H, x:x + w]
    pts = getattr(state, stream)
    if pts is not None:
        draw_hand(roi, pts, state.side,
                  (w / 2.0, HAND_H - WRIST_FROM_BOTTOM),
                  colour if seen else GREY_BGR, mm_per_px)
    if not seen:
        label = "no hand"
        put(img, label, (x + (w - text_width(label, 0.9, 2)) // 2,
                         top + HAND_H // 2), 0.9, TEXT_BGR, 2)


def _guess_block(img, x: int, y: int, guess: Optional[Guess],
                 truth: Optional[str], note: str) -> None:
    if guess is None:
        put(img, "pose guess: " + (note or "none"), (x, y + 24), 0.6, DIM_BGR)
        return
    end = put_segments(img, [("pose guess: ", DIM_BGR),
                             (guess.pose, TEXT_BGR)], (x, y + 26), 0.8, 2)
    if truth is not None:
        draw_tick(img, end + 14, y + 26, guess.pose == truth, 18)
    if guess.margin is not None:
        put(img, f"margin {guess.margin:.2f} to {guess.runner_up} "
                 f"(distance {guess.distance:.2f})", (x, y + 52), 0.5, DIM_BGR)
    if truth is not None:
        put_segments(img, [("true pose: ", DIM_BGR), (truth, TEXT_BGR)],
                     (x, y + 78), 0.6)
    if note:
        put(img, note, (x, y + 100), 0.45, DIM_BGR)


def render(states: Mapping[str, SideState], sides: Sequence[str], now: float,
           header: Sequence[str] = (), footer: str = "",
           guesses: Optional[Mapping[str, Optional[Guess]]] = None,
           truth: Optional[str] = None, guess_note: str = "",
           progress: Optional[float] = None,
           mm_per_px: float = MM_PER_PX) -> np.ndarray:
    """The whole picture: one row per side in `sides`, each with the glove,
    camera and fused hands, the fused hand's badges and the pose guess.

    `header` is up to two lines at the top, `footer` the key help;
    `guesses` per side (see `CentroidClassifier.guess`), `truth` the true
    pose when it is known (a replay), `progress` 0..1 of the current take.
    """
    w, h = canvas_size(len(sides))
    img = np.full((h, w, 3), BG_BGR, np.uint8)
    title = "GLOVE + CAMERA FUSION"
    put(img, title, (MARGIN, 30), 0.85, TEXT_BGR, 2)
    x = MARGIN + text_width(title, 0.85, 2) + 24
    for stream in STREAMS:
        cv2.circle(img, (x + 7, 24), 7, STREAM_BGR[stream], -1, cv2.LINE_AA)
        x = put(img, stream, (x + 20, 30), 0.6, TEXT_BGR) + 18
    header = list(header)[:2]
    if header:
        put(img, header[0], (MARGIN, 54), 0.5, TEXT_BGR)
    if len(header) > 1:
        put(img, header[1], (w - MARGIN - text_width(header[1]), 54), 0.5,
            DIM_BGR)
    if progress is not None:
        y = HEADER_H - 8
        cv2.rectangle(img, (MARGIN, y), (w - MARGIN, y + 3), PANEL_BGR, -1)
        cv2.rectangle(img, (MARGIN, y),
                      (MARGIN + int((w - 2 * MARGIN)
                                    * min(1.0, max(0.0, progress))), y + 3),
                      DIM_BGR, -1)

    for r, side in enumerate(sides):
        state = states.get(side) or SideState(side)
        y0 = HEADER_H + r * ROW_H
        if r:
            cv2.line(img, (MARGIN, y0 - 4), (w - MARGIN, y0 - 4), PANEL_BGR, 1)
        y1 = y0
        for k, stream in enumerate(STREAMS):
            _panel(img, state, stream, _col_x(k), y1, COL_W[k], now, mm_per_px,
                   prefix=f"{side.upper()} HAND" if k == 0 else "")
        y2 = y1 + PANEL_TITLE_H + HAND_H + 8
        guess = (guesses or {}).get(side)
        if not state.seen("fused", now):
            guess = None
        _guess_block(img, _col_x(0) + 4, y2, guess, truth, guess_note)
        bx = _col_x(2) + 4
        frame = state.frame
        if frame is not None and state.seen("fused", now):
            for i, line in enumerate(badge_cells(frame.dof_source,
                                                 frame.rail_active)):
                for col, cell in zip(BADGE_COLS, line):
                    put_segments(img, cell, (bx + col, y2 + 16 + i * LINE_H),
                                 0.5)
            put(img, status_text(state.camera_fresh(now), state.glove_hz(now)),
                (bx, y2 + 16 + 6 * LINE_H), 0.5,
                TEXT_BGR if state.camera_fresh(now) else BAD_BGR)
        else:
            put(img, "no fused frame", (bx, y2 + 16), 0.5, DIM_BGR)
    if footer:
        put(img, footer, (MARGIN, h - 10), 0.5, DIM_BGR)
    return img


def render_message(lines: Sequence[str], n_rows: int = 1) -> np.ndarray:
    """A plain screen of text (loading, the warm-up), the demo's size."""
    w, h = canvas_size(n_rows)
    img = np.full((h, w, 3), BG_BGR, np.uint8)
    put(img, "GLOVE + CAMERA FUSION", (MARGIN, 30), 0.85, TEXT_BGR, 2)
    for i, line in enumerate(lines):
        put(img, line, (MARGIN, 90 + i * 34), 0.7 if i == 0 else 0.6,
            TEXT_BGR if i == 0 else DIM_BGR, 2 if i == 0 else 1)
    return img
