"""Skeleton overlay + HUD drawing on OpenCV frames.

Colour convention matches the glove viewer (xr_hand.viz3d): cyan = left,
red = right, so live camera and glove playback read the same at a glance.

Text is always drawn on the final (possibly mirrored) display frame so it
stays readable; geometry is drawn pre-mirror so it stays glued to the hand.
"""
import cv2

# Bone connectivity for the 21-keypoint MediaPipe layout.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # wrist -> pinky base (palm edge)
]

LEFT_BGR = (255, 255, 0)    # cyan
RIGHT_BGR = (0, 0, 255)     # red
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Outline drawn as eight offset copies at the SAME thickness as the fill.
# Drawing a thick black pass under a thin coloured pass looks like an outline
# but is not one: OpenCV's Hershey glyphs get wider as thickness grows, so the
# two passes start together and drift apart by the end of the string, which
# reads as doubled text. Same thickness everywhere means identical widths.
_OUTLINE_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0),
                    (1, 0), (-1, 1), (0, 1), (1, 1))


def hand_color(side: str):
    return LEFT_BGR if side == "left" else RIGHT_BGR


def draw_text(frame, text: str, org, scale: float, color, thickness: int = 1,
              outline: int = 2) -> None:
    """Text with a readable dark outline, drift-free."""
    x, y = int(org[0]), int(org[1])
    if outline > 0:
        for dx, dy in _OUTLINE_OFFSETS:
            cv2.putText(frame, text, (x + dx * outline, y + dy * outline),
                        FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def text_size(text: str, scale: float, thickness: int):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def _fit_scale(frame, text: str, scale: float, thickness: int,
               margin: int = 24) -> float:
    """Shrink the scale so a long pose name still fits the frame width."""
    avail = frame.shape[1] - 2 * margin
    w, _ = text_size(text, scale, thickness)
    return scale * avail / w if w > avail > 0 else scale


def _centered_x(frame, text: str, scale: float, thickness: int) -> int:
    w, _ = text_size(text, scale, thickness)
    return max(6, (frame.shape[1] - w) // 2)


def draw_hand(frame, hand) -> None:
    """Draw one CamHand's skeleton (image-space landmarks) on the frame."""
    color = hand_color(hand.hand_side)
    pts = [(int(round(p[0])), int(round(p[1]))) for p in hand.img]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 5 if i in (0, 4, 8, 12, 16, 20) else 3   # wrist + fingertips bigger
        cv2.circle(frame, p, r, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, color, 1, cv2.LINE_AA)


def label_hands(frame, hands, mirrored: bool) -> None:
    """Side + confidence next to each wrist. Call on the display frame."""
    w = frame.shape[1]
    for hand in hands:
        x, y = hand.img[0][0], hand.img[0][1]
        if mirrored:
            x = w - x
        text = f"{hand.hand_side} {hand.score:.2f}"
        tw, _th = text_size(text, 0.6, 1)
        # centre the label under the wrist and keep it inside the frame
        px = min(max(6, int(x) - tw // 2), frame.shape[1] - tw - 6)
        py = min(int(y) + 28, frame.shape[0] - 8)
        draw_text(frame, text, (px, py), 0.6, hand_color(hand.hand_side))


def draw_hud(frame, lines) -> None:
    """Small status lines, top-left."""
    y = 24
    for line in lines:
        draw_text(frame, line, (10, y), 0.55, (80, 255, 80))
        y += 24


def draw_banner(frame, text: str, sub: str = "", rec: bool = False) -> None:
    """Big centred prompt for the guided recorder (pose name, countdown).

    Both lines are measured before they are placed, so they stay centred for
    any pose name, and a long one is scaled down rather than running off the
    edge of the frame.
    """
    h, w = frame.shape[:2]

    scale = _fit_scale(frame, text, 1.2, 2)
    draw_text(frame, text, (_centered_x(frame, text, scale, 2), h // 2 - 18),
              scale, WHITE, thickness=2, outline=2)

    if sub:
        sub_scale = _fit_scale(frame, sub, 0.7, 1)
        draw_text(frame, sub, (_centered_x(frame, sub, sub_scale, 1), h // 2 + 22),
                  sub_scale, (0, 255, 255))

    if rec:
        rec_w, _ = text_size("REC", 0.7, 2)
        cv2.circle(frame, (w - 26, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)
        draw_text(frame, "REC", (w - 48 - rec_w, 38), 0.7, (0, 0, 255),
                  thickness=2)
