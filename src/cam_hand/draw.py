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


def hand_color(side: str):
    return LEFT_BGR if side == "left" else RIGHT_BGR


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
        org = (int(x) - 40, int(y) + 28)
        cv2.putText(frame, text, org, FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, FONT, 0.6, hand_color(hand.hand_side), 1,
                    cv2.LINE_AA)


def draw_hud(frame, lines) -> None:
    """Small status lines, top-left."""
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), FONT, 0.55, (80, 255, 80), 1, cv2.LINE_AA)
        y += 22


def draw_banner(frame, text: str, sub: str = "", rec: bool = False) -> None:
    """Big centred prompt for the guided recorder (pose name, countdown)."""
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (max(10, w // 2 - 10 * len(text)), h // 2 - 20),
                FONT, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, (max(10, w // 2 - 10 * len(text)), h // 2 - 20),
                FONT, 1.2, WHITE, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (max(10, w // 2 - 7 * len(sub)), h // 2 + 20),
                    FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, sub, (max(10, w // 2 - 7 * len(sub)), h // 2 + 20),
                    FONT, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    if rec:
        cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "REC", (w - 85, 38), FONT, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
