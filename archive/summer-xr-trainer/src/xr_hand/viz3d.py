"""Interactive 3D skeletal viewer.

Built on matplotlib's mplot3d. The HandViewer class is the only surface the
rest of the pipeline touches, so you can later swap matplotlib for pyvista /
open3d / a Unity bridge without changing scripts.

Styling matches the StretchSense glove vibe: dark background, cyan left
hand, red right hand, highlighted fingertips, semi-transparent palm fill.
"""
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

from .joints import BONES, HandFrame, JOINT_COUNT, JOINT_INDEX
from .kinematics import forward_kinematics


FINGERTIP_IDX = [
    JOINT_INDEX[n] for n in
    ("THUMB_TIP", "INDEX_TIP", "MIDDLE_TIP", "RING_TIP", "LITTLE_TIP")
]

# Vertices for the palm polygon, walking around the back of the hand
PALM_POLYGON_IDX = [
    JOINT_INDEX["WRIST"],
    JOINT_INDEX["THUMB_METACARPAL"],
    JOINT_INDEX["INDEX_METACARPAL"],
    JOINT_INDEX["MIDDLE_METACARPAL"],
    JOINT_INDEX["RING_METACARPAL"],
    JOINT_INDEX["LITTLE_METACARPAL"],
]

_HAND_COLORS = {
    "left":  {"bone": "#22e0e0", "joint": "#22e0e0", "tip": "#a0ffff", "palm": "#08303a"},
    "right": {"bone": "#ff5b5b", "joint": "#ff5b5b", "tip": "#ffc0c0", "palm": "#400a14"},
}

# Where each hand's wrist is pinned on screen (metres). With the default
# camera (azim=90) +x lands on the viewer's LEFT, so left gets +x and right
# gets -x — left hand on the left of the screen, as you'd see your own hands.
_HAND_ANCHOR_X = {"left": 0.12, "right": -0.12}


class HandViewer:
    def __init__(self, title: str = "XR Hand", view_radius: float = 0.22):
        self.fig = plt.figure(figsize=(8, 7), facecolor="black")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="black")
        self.ax.set_title(title, color="white", pad=12)

        # Hide tick labels for a clean look; keep panes subtle for depth cue.
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.ax.tick_params(colors="#202020", length=0)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.04, 0.04, 0.07, 1.0))
            axis.line.set_color("#1a1a1a")
            axis.label.set_color("#202020")

        r = view_radius
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_zlim(-r, r)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        # Default camera: look at the back of the hand with fingers pointing
        # up, thumb on the LEFT (matches a left glove laid flat in the
        # StretchSense product shot).
        self.ax.view_init(elev=15, azim=90)

        self._scatters: Dict[str, Path3DCollection] = {}
        self._tip_scatters: Dict[str, Path3DCollection] = {}
        self._bones: Dict[str, list] = {}
        self._palms: Dict[str, Poly3DCollection] = {}
        self._anim: Optional[FuncAnimation] = None

    def _ensure_hand(self, hand: str) -> None:
        if hand in self._scatters:
            return
        c = _HAND_COLORS.get(hand, _HAND_COLORS["left"])
        self._scatters[hand] = self.ax.scatter(
            [0] * JOINT_COUNT, [0] * JOINT_COUNT, [0] * JOINT_COUNT,
            c=c["joint"], s=22, label=hand, edgecolors="none", depthshade=False,
        )
        self._tip_scatters[hand] = self.ax.scatter(
            [0] * len(FINGERTIP_IDX), [0] * len(FINGERTIP_IDX), [0] * len(FINGERTIP_IDX),
            c=c["tip"], s=90, edgecolors="white", linewidths=0.6, depthshade=False,
        )
        lines = []
        for _ in BONES:
            (ln,) = self.ax.plot([0, 0], [0, 0], [0, 0], color=c["bone"], lw=2.2)
            lines.append(ln)
        self._bones[hand] = lines

        palm = Poly3DCollection(
            [[(0, 0, 0)] * len(PALM_POLYGON_IDX)],
            facecolors=c["palm"], edgecolors=c["bone"], linewidths=0.8, alpha=0.55,
        )
        self.ax.add_collection3d(palm)
        self._palms[hand] = palm

        leg = self.ax.legend(loc="upper right", facecolor="black", edgecolor="#303030")
        for text in leg.get_texts():
            text.set_color("white")

    def update_hand(self, frame: HandFrame) -> None:
        hand = frame.hand_side
        self._ensure_hand(hand)
        positions = forward_kinematics(frame)
        # Re-anchor the wrist to a fixed per-hand x: the real glove streams no
        # world position (both roots arrive at the origin), so without this
        # left and right draw on top of each other. Display-only — recorded
        # frames keep the stream's own coordinates.
        wx, wy, wz = positions[JOINT_INDEX["WRIST"]]
        anchor_x = _HAND_ANCHOR_X.get(hand, 0.0)
        # Flip Z so fingers point up in matplotlib's default coordinate system.
        xs = [p[0] - wx + anchor_x for p in positions]
        ys = [p[1] - wy for p in positions]
        zs = [wz - p[2] for p in positions]

        self._scatters[hand]._offsets3d = (xs, ys, zs)
        self._tip_scatters[hand]._offsets3d = (
            [xs[i] for i in FINGERTIP_IDX],
            [ys[i] for i in FINGERTIP_IDX],
            [zs[i] for i in FINGERTIP_IDX],
        )
        for ln, (a, b) in zip(self._bones[hand], BONES):
            ln.set_data_3d([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]])

        palm_verts = [(xs[i], ys[i], zs[i]) for i in PALM_POLYGON_IDX]
        self._palms[hand].set_verts([palm_verts])

    def run(self, on_tick: Callable[[], None], interval_ms: int = 33) -> None:
        def _update(_):
            on_tick()
            return ()
        self._anim = FuncAnimation(
            self.fig, _update, interval=interval_ms, blit=False,
            cache_frame_data=False,
        )
        plt.show()
