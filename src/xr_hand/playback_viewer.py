"""Scrubbable playback viewer: 3D hand on the left, per-joint data on the right.

Unlike the live viewer, this loads a whole recording up front and lets you
move through it with a slider (or the left/right arrow keys), like scrubbing a
video. The data panel on the right always shows the joint values for the frame
you're parked on, and the joint that moved the most into this frame is
highlighted — so the numbers stay aligned with whatever the hand is doing.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RangeSlider, Slider
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

from .export_xlsx import write_workbook
from .joints import BONES, HandFrame, JOINT_COUNT, JOINT_NAMES
from .kinematics import forward_kinematics
from .viz3d import _HAND_COLORS, FINGERTIP_IDX, PALM_POLYGON_IDX


class PlaybackViewer:
    def __init__(
        self,
        frames_with_time: List[Tuple[HandFrame, float]],
        title: str = "Playback",
        view_radius: float = 0.22,
        source_path: Optional[Path] = None,
    ):
        if not frames_with_time:
            raise ValueError("no frames to play back")

        self.frames = [f for f, _ in frames_with_time]
        self.times = [t for _, t in frames_with_time]
        self.n = len(self.frames)
        self.t0 = self.times[0]
        self.source_path = Path(source_path) if source_path else None
        # selected window [lo, hi] (frame indices); starts as the whole clip
        self._lo = 0
        self._hi = self.n - 1

        # For any scrub index i, which is the latest left/right frame at or
        # before i? (so both hands stay on screen even though packets alternate)
        self._last_left = self._build_last_index("left")
        self._last_right = self._build_last_index("right")
        # Previous frame of the *same* hand, for the movement highlight.
        self._prev_same = self._build_prev_same()
        # Cache world positions per frame (computed lazily).
        self._fk_cache: List[Optional[list]] = [None] * self.n

        self._build_figure(title, view_radius)
        self._playing = False
        self._timer = self.fig.canvas.new_timer(interval=self._play_interval_ms())
        self._timer.add_callback(self._on_play_tick)

        self._draw_index(0)
        self._on_range((self._lo, self._hi))

    # --- precompute helpers -------------------------------------------
    def _build_last_index(self, hand: str) -> List[Optional[int]]:
        out: List[Optional[int]] = [None] * self.n
        last: Optional[int] = None
        for i, f in enumerate(self.frames):
            if f.hand_side == hand:
                last = i
            out[i] = last
        return out

    def _build_prev_same(self) -> List[Optional[int]]:
        out: List[Optional[int]] = [None] * self.n
        seen: dict[str, int] = {}
        for i, f in enumerate(self.frames):
            out[i] = seen.get(f.hand_side)
            seen[f.hand_side] = i
        return out

    def _fk(self, i: int) -> list:
        if self._fk_cache[i] is None:
            self._fk_cache[i] = forward_kinematics(self.frames[i])
        return self._fk_cache[i]

    def _play_interval_ms(self) -> int:
        if self.n < 2:
            return 200
        dts = [b - a for a, b in zip(self.times, self.times[1:]) if b > a]
        if not dts:
            return 200
        dts.sort()
        median = dts[len(dts) // 2]
        return max(20, min(1000, int(median * 1000)))

    # --- figure construction ------------------------------------------
    def _build_figure(self, title: str, view_radius: float) -> None:
        self.fig = plt.figure(figsize=(15.5, 8.5), facecolor="black")
        self.fig.suptitle(title, color="white", fontsize=12)
        gs = GridSpec(
            1, 2, figure=self.fig,
            width_ratios=[1.0, 1.3],
            left=0.02, right=0.98, top=0.92, bottom=0.26, wspace=0.05,
        )

        # 3D hand
        self.ax = self.fig.add_subplot(gs[0, 0], projection="3d", facecolor="black")
        self.ax.set_xticklabels([]); self.ax.set_yticklabels([]); self.ax.set_zticklabels([])
        self.ax.tick_params(colors="#202020", length=0)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.04, 0.04, 0.07, 1.0))
            axis.line.set_color("#1a1a1a")
        r = view_radius
        self.ax.set_xlim(-r, r); self.ax.set_ylim(-r, r); self.ax.set_zlim(-r, r)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        self.ax.view_init(elev=15, azim=90)

        # Data panel (text rows)
        self.tax = self.fig.add_subplot(gs[0, 1], facecolor="#0a0a0f")
        self.tax.set_xlim(0, 1); self.tax.set_ylim(0, 1)
        self.tax.axis("off")
        self._clock = self.tax.text(
            0.0, 0.99, "", color="#22e0e0", family="monospace",
            fontsize=14, fontweight="bold", va="top", ha="left",
            transform=self.tax.transAxes,
        )
        self._header = self.tax.text(
            0.0, 0.94, "", color="white", family="monospace",
            fontsize=10, va="top", ha="left", transform=self.tax.transAxes,
        )
        # Two hand sections share one joint-name column on the left.
        self._LEFT_X = 0.36
        self._RIGHT_X = 0.69
        xyz_head = f"{'X':>7}{'Y':>7}{'Z':>7}"
        self.tax.text(
            self._LEFT_X, 0.90, "LEFT", color=_HAND_COLORS["left"]["bone"],
            family="monospace", fontsize=10, fontweight="bold",
            va="top", ha="left", transform=self.tax.transAxes,
        )
        self.tax.text(
            self._RIGHT_X, 0.90, "RIGHT", color=_HAND_COLORS["right"]["bone"],
            family="monospace", fontsize=10, fontweight="bold",
            va="top", ha="left", transform=self.tax.transAxes,
        )
        self.tax.text(
            0.0, 0.875, "JOINT", color="#7fd7ff", family="monospace",
            fontsize=8, va="top", ha="left", transform=self.tax.transAxes,
        )
        self.tax.text(
            self._LEFT_X, 0.875, xyz_head, color="#7fd7ff", family="monospace",
            fontsize=8, va="top", ha="left", transform=self.tax.transAxes,
        )
        self.tax.text(
            self._RIGHT_X, 0.875, xyz_head, color="#7fd7ff", family="monospace",
            fontsize=8, va="top", ha="left", transform=self.tax.transAxes,
        )
        self._name_rows = []
        self._left_rows = []
        self._right_rows = []
        top = 0.85
        step = top / (JOINT_COUNT + 1)
        for k in range(JOINT_COUNT):
            y = top - (k + 1) * step
            self._name_rows.append(self.tax.text(
                0.0, y, "", color="#bbbbbb", family="monospace",
                fontsize=8, va="center", ha="left", transform=self.tax.transAxes,
            ))
            self._left_rows.append(self.tax.text(
                self._LEFT_X, y, "", color="#dddddd", family="monospace",
                fontsize=8, va="center", ha="left", transform=self.tax.transAxes,
            ))
            self._right_rows.append(self.tax.text(
                self._RIGHT_X, y, "", color="#dddddd", family="monospace",
                fontsize=8, va="center", ha="left", transform=self.tax.transAxes,
            ))

        rmax = max(1, self.n - 1)

        # Scrub slider (move through frames one at a time)
        self.sax = self.fig.add_axes([0.10, 0.185, 0.82, 0.025], facecolor="#202028")
        self.slider = Slider(
            self.sax, "Frame", 0, rmax, valinit=0, valstep=1, color="#22e0e0",
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        # Range slider (select a window for play + export)
        self.rax = self.fig.add_axes([0.10, 0.125, 0.82, 0.025], facecolor="#202028")
        self.range = RangeSlider(
            self.rax, "Select", 0, rmax, valinit=(0, rmax), valstep=1,
            color="#ffe14d",
        )
        self.range.label.set_color("#ffe14d")
        self.range.valtext.set_color("#ffe14d")
        self.range.on_changed(self._on_range)

        # Play / pause button
        self.bax = self.fig.add_axes([0.10, 0.04, 0.08, 0.05])
        self.play_btn = Button(self.bax, "Play", color="#2a2a35", hovercolor="#3a3a48")
        self.play_btn.label.set_color("white")
        self.play_btn.on_clicked(self._on_play_clicked)

        # Export selection button
        self.eax = self.fig.add_axes([0.19, 0.04, 0.16, 0.05])
        self.export_btn = Button(
            self.eax, "Export selection → Excel",
            color="#2a3a2a", hovercolor="#3a4a3a",
        )
        self.export_btn.label.set_color("white")
        self.export_btn.on_clicked(self._on_export)

        # Selection / status readout
        self._sel_info = self.fig.text(
            0.37, 0.065, "", color="#ffe14d", family="monospace",
            fontsize=10, va="center", ha="left",
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._scatters: dict[str, Path3DCollection] = {}
        self._tip_scatters: dict[str, Path3DCollection] = {}
        self._bones: dict[str, list] = {}
        self._palms: dict[str, Poly3DCollection] = {}

    def _ensure_hand(self, hand: str) -> None:
        if hand in self._scatters:
            return
        c = _HAND_COLORS.get(hand, _HAND_COLORS["left"])
        self._scatters[hand] = self.ax.scatter(
            [0] * JOINT_COUNT, [0] * JOINT_COUNT, [0] * JOINT_COUNT,
            c=c["joint"], s=22, edgecolors="none", depthshade=False,
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

    def _draw_hand(self, i: int) -> None:
        hand = self.frames[i].hand_side
        self._ensure_hand(hand)
        positions = self._fk(i)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [-p[2] for p in positions]
        self._scatters[hand]._offsets3d = (xs, ys, zs)
        self._tip_scatters[hand]._offsets3d = (
            [xs[t] for t in FINGERTIP_IDX],
            [ys[t] for t in FINGERTIP_IDX],
            [zs[t] for t in FINGERTIP_IDX],
        )
        for ln, (a, b) in zip(self._bones[hand], BONES):
            ln.set_data_3d([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]])
        self._palms[hand].set_verts([[(xs[k], ys[k], zs[k]) for k in PALM_POLYGON_IDX]])

    def _moved_joints(self, i: int) -> set:
        """All joints that moved into frame i (not just the single biggest).

        A joint counts as moving if it travelled past a small noise floor AND a
        low fraction of this frame's peak motion. The relative band keeps minor
        joint coupling from lighting up the whole hand, while the low fraction
        ensures a second or third finger moving at once is still caught.
        """
        prev = self._prev_same[i]
        if prev is None:
            return set()
        cur = self._fk(i)
        old = self._fk(prev)
        dists = []
        for k in range(min(len(cur), len(old))):
            dx = cur[k][0] - old[k][0]
            dy = cur[k][1] - old[k][1]
            dz = cur[k][2] - old[k][2]
            dists.append((dx * dx + dy * dy + dz * dz) ** 0.5)
        if not dists:
            return set()
        maxd = max(dists)
        floor = 0.003  # ~3 mm; below this the hand is essentially still
        if maxd < floor:
            return set()
        thresh = max(floor, 0.15 * maxd)
        return {k for k, d in enumerate(dists) if d >= thresh}

    def _fill_section(self, rows, src_idx: Optional[int]) -> set:
        """Fill one hand's value column from frame src_idx; return its moved set."""
        if src_idx is None:
            for txt in rows:
                txt.set_text("     --     --     --")
                txt.set_color("#555555")
                txt.set_fontweight("normal")
            return set()
        frame = self.frames[src_idx]
        world = self._fk(src_idx)  # world positions move with the hand; local don't
        hot = self._moved_joints(src_idx)
        for k in range(JOINT_COUNT):
            if k < len(world):
                wx, wy, wz = world[k]
                rows[k].set_text(f"{wx:>7.3f}{wy:>7.3f}{wz:>7.3f}")
            else:
                rows[k].set_text("")
            if k in hot:
                rows[k].set_color("#ffe14d")
                rows[k].set_fontweight("bold")
            else:
                rows[k].set_color("#dddddd")
                rows[k].set_fontweight("normal")
        return hot

    def _update_panel(self, i: int) -> None:
        t = self.times[i] - self.t0
        clock = datetime.fromtimestamp(self.times[i]).strftime("%H:%M:%S.") + \
            f"{int((self.times[i] % 1) * 1000):03d}"
        self._clock.set_text(f"CLOCK  {clock}")
        self._header.set_text(f"frame {i + 1}/{self.n}   +{t:6.3f} s")

        left_hot = self._fill_section(self._left_rows, self._last_left[i])
        right_hot = self._fill_section(self._right_rows, self._last_right[i])
        moved = left_hot | right_hot
        for k in range(JOINT_COUNT):
            self._name_rows[k].set_text(JOINT_NAMES[k] if k < len(JOINT_NAMES) else "")
            self._name_rows[k].set_color("#ffe14d" if k in moved else "#bbbbbb")

    def _draw_index(self, i: int) -> None:
        i = max(0, min(self.n - 1, int(i)))
        # draw both hands at their most recent pose up to i
        li = self._last_left[i]
        ri = self._last_right[i]
        if li is not None:
            self._draw_hand(li)
        if ri is not None:
            self._draw_hand(ri)
        self._update_panel(i)
        self.fig.canvas.draw_idle()

    # --- callbacks -----------------------------------------------------
    def _on_slider(self, val) -> None:
        self._draw_index(int(val))

    def _on_key(self, event) -> None:
        if event.key in ("right", "left"):
            cur = int(self.slider.val)
            nxt = cur + (1 if event.key == "right" else -1)
            self.slider.set_val(max(0, min(self.n - 1, nxt)))

    def _on_play_clicked(self, _event) -> None:
        self._playing = not self._playing
        self.play_btn.label.set_text("Pause" if self._playing else "Play")
        if self._playing:
            # start from the window's beginning if we're outside it
            cur = int(self.slider.val)
            if cur < self._lo or cur >= self._hi:
                self.slider.set_val(self._lo)
            self._timer.start()
        else:
            self._timer.stop()

    def _on_play_tick(self) -> None:
        cur = int(self.slider.val)
        if cur >= self._hi:  # stop at the end of the selected window
            self._playing = False
            self.play_btn.label.set_text("Play")
            self._timer.stop()
            return
        self.slider.set_val(cur + 1)

    # --- time-range selection + export --------------------------------
    def _fmt_elapsed(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        return f"{int(m)}:{s:06.3f}"

    def _fmt_clock(self, wall: float) -> str:
        return datetime.fromtimestamp(wall).strftime("%H:%M:%S.") + \
            f"{int((wall % 1) * 1000):03d}"

    def _on_range(self, val) -> None:
        self._lo, self._hi = int(val[0]), int(val[1])
        lo_el = self._fmt_elapsed(self.times[self._lo] - self.t0)
        hi_el = self._fmt_elapsed(self.times[self._hi] - self.t0)
        nframes = self._hi - self._lo + 1
        self._sel_info.set_text(
            f"window  {lo_el}-{hi_el}   "
            f"(clock {self._fmt_clock(self.times[self._lo])} - "
            f"{self._fmt_clock(self.times[self._hi])})   {nframes} frames"
        )
        self.fig.canvas.draw_idle()

    def _export_path(self) -> Path:
        lo_s = self.times[self._lo] - self.t0
        hi_s = self.times[self._hi] - self.t0
        tag = f"_{lo_s:.1f}-{hi_s:.1f}s"
        if self.source_path is not None:
            return self.source_path.with_name(self.source_path.stem + tag + ".xlsx")
        return Path("recordings") / f"selection{tag}.xlsx"

    def _export_rows(self):
        """One tidy row per (frame x hand x joint) over the selected window."""
        for i in range(self._lo, self._hi + 1):
            fr = self.frames[i]
            world = self._fk(i)
            hot = self._moved_joints(i)
            elapsed = self.times[i] - self.t0
            clock = self._fmt_clock(self.times[i])
            for k in range(JOINT_COUNT):
                wx, wy, wz = world[k] if k < len(world) else (0.0, 0.0, 0.0)
                j = fr.joints[k] if k < len(fr.joints) else None
                yield [
                    i + 1, round(elapsed, 3), clock, fr.hand_side, JOINT_NAMES[k],
                    wx, wy, wz,
                    j.qw if j else 0.0, j.qx if j else 0.0,
                    j.qy if j else 0.0, j.qz if j else 0.0,
                    "yes" if k in hot else "",
                ]

    def _on_export(self, _event) -> None:
        out = self._export_path()
        nframes = self._hi - self._lo + 1
        summary = [
            ("Source", self.source_path.name if self.source_path else "(live)"),
            ("Window (elapsed)",
             f"{self._fmt_elapsed(self.times[self._lo] - self.t0)} - "
             f"{self._fmt_elapsed(self.times[self._hi] - self.t0)}"),
            ("Window (clock)",
             f"{self._fmt_clock(self.times[self._lo])} - "
             f"{self._fmt_clock(self.times[self._hi])}"),
            ("Frames", nframes),
            ("Joints per hand", JOINT_COUNT),
            ("Positions", "world space (metres)"),
        ]
        write_workbook(out, summary=summary, rows=self._export_rows())
        self._sel_info.set_text(f"saved {nframes} frames -> {out.name}")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()
