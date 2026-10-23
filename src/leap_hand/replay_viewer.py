"""Replay viewer: one recorded hand, on the camera's own clock, as recorded.

Built on the repository's matplotlib playback stack (`xr_hand.playback_viewer`,
`xr_hand.viz3d`): dark figure, a 3D axes, Slider and Button widgets, key
bindings, the same hand colours. Every number it shows comes from
`leap_hand.replay_core` (poses, clock, events, fidelity) or `leap_hand.paper3`
(the research comparison), so each can be reproduced from a Python prompt.
Works on the Agg backend without a window (tests, headless screenshots);
`show()` runs the GUI loop (TkAgg on this machine).

SKELETON ONLY
  The 26 reconstructed joints, their bones, fingertips and the palm polygon.
  No skinned mesh: a mesh is a retargeting of the skeleton onto a model, and
  any such retargeting would live apart from the source skeleton, not in the
  viewer whose job is to show what the tracker recorded.

WHAT IS DRAWN
  skeleton   `replay_core.reconstruct` of the displayed source frame, or the
             `DisplayedPose` of `pose_at` in interpolate mode.
  overlay    (o) the recorder's own `abs26` positions as hollow markers and a
             red segment from each reconstructed joint to its recorded
             position, to scale (the residuals are below 1 um, so the
             segments are invisible; the numbers are in the HUD). Hidden
             between source frames: an interpolated pose has no recording.
  axes       (a) a 1 cm triad per joint from the columns of its rotation:
             x red, y green, z blue.
  reference  (r) camera: the recorded motion in LeapC desktop space, axis
             limits fixed once from the bounding box of every reconstructed
             joint of the recording (equal aspect, never re-centred per
             frame); wrist / palm: `replay_core.to_reference`, limits centred
             on the origin. The plot axes are the frame's own axes with y
             vertical (`view_init(vertical_axis='y')`), first seen from +z:
             camera +x right, +y up, +z toward the viewer, a proper rotation,
             so a left hand stays a left hand. The wrist is never re-anchored.

TWO PANELS, KEPT APART
  HUD        replay fidelity: what is shown, from which line, on which clock,
             and how far the reconstruction is from the recording itself.
  Paper 3    the research comparison (`paper3.frame_relation_values`), with
             the class, the MCP-order disclosure and the six angles the
             camera does not give. Not a fidelity number.
  Both are monospace text rasterised with Pillow (`_TextPanel`, why there).

CONTROLS (buttons and keys)
  space play/pause   +/- speed 0.1 0.25 0.5 1 2 4   slider: scrub by source
  frame   left/right step one source frame   p/n previous/next event
  r reference   i hold/interpolate   o overlay   a axes   s screenshot   q/esc
  quit. Default mode is hold: source frames only, never interpolated unless
  switched on. Stepping and event jumps pause; they and the slider land on a
  frame's exact time ("recorded").

PLAYBACK TIMING (TIMING_DOC says exactly what is logged)
  Time-driven: a timer (TICK_MS) calls `tick()`, which reads
  `PlaybackClock.media_time` and shows `pose_at(rec, t, mode)`, so a late tick
  shows a later frame instead of stretching time. Each tick logs its lateness
  against its timer slot to a `SchedulingLog`; the HUD and the timing log
  (`write_timing_log`) carry the statistics. Playback stops at the end.
"""
import json
import textwrap
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image, ImageDraw, ImageFont

from xr_hand.joints import BONES
from xr_hand.viz3d import _HAND_COLORS, FINGERTIP_IDX, PALM_POLYGON_IDX

from . import paper3 as p3
from . import replay_core as rc

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
TICK_MS = 15
TRIAD_M = 0.01                      # joint axes length, 1 cm
WRITER_STALL_S = 0.05               # HUD flags a writer jump this much over the camera's

REFERENCE_LABELS = {
    "camera": "camera space (LeapC desktop): +x right, +y up, +z toward operator, metres",
    "wrist": ("wrist-relative: origin WRIST joint, axes = WRIST orientation "
              "(LeapC forearm on the camera)"),
    "palm": "palm-relative: origin WRIST joint, axes = PALM orientation",
}
PLOT_MAPPING = {
    "camera": ("plot axes = camera axes, y vertical, first seen from the "
               "operator's side (+z) 30 deg above; limits fixed for the recording"),
    "wrist": ("plot axes = WRIST axes (LeapC +y dorsal, -z distal), y vertical, "
              "first seen from +z+x 25 deg above; limits centred on the origin"),
    "palm": ("plot axes = PALM axes (LeapC +y dorsal, -z distal), y vertical, "
             "first seen from +z+x 25 deg above; limits centred on the origin"),
}
VIEWS = {"camera": (30, 0), "wrist": (25, 60), "palm": (25, 60)}   # elev, azim
CLOCK_OK = "clock: timestamp_us (camera acquisition, no fallback)"
FID_POSITION = "FK vs recorded abs26"
FID_CHAIN = "matrix chain vs quaternion chain (same data, two code paths)"
FID_BONE = "bone -z axis vs recorded bone vector"
P3_TITLE = "Paper 3 agreement (research model, not fidelity)"
RECORDED_NA = "recorded: n/a between source frames"
TIMING_DOC = (
    "One record per tick while playing. intended = the media time the tick "
    "was due at had every tick taken no time: the previous tick's media time "
    "(or the media time at play, a seek or a speed change) plus one timer "
    "interval x speed. actual = the media time the PlaybackClock had reached "
    "when the tick ran; that time chooses the pose. |actual - intended| is "
    "the tick's lateness in media-time ms (divide by the speed for wall ms): "
    "the previous tick's drawing, the GUI and the OS all show up there. "
    "frames_skipped = source frames between consecutive displayed frames "
    "inside one uninterrupted play run; a seek, a pause or a restart begins "
    "a new run, so a jump is not counted as skipped frames.")

C_BG = "black"
C_PANEL = "#0a0a0f"
C_P3_PANEL = "#120c18"
C_TEXT = "#dddddd"
C_DIM = "#8a8a99"
C_TITLE = "#ffe14d"
C_CYAN = "#7fd7ff"
C_GOOD = "#5fe08a"
C_WARN = "#ffb000"
C_BAD = "#ff5050"
C_P3_TITLE = "#c9a2ff"
C_P3_EDGE = "#7a4fb0"
STATUS_COLORS = {"applied": "#f0f0f0", "informational": C_CYAN,
                 "not_applied": "#7a7a85", "unavailable": "#5c5c6c"}
EVENT_COLORS = {"gap": "#ff3b3b", "id_change": "#ff9f1a",
                "dropped_frames": "#ffe14d", "invalid_status": "#9a9a9a",
                "clock_defect": "#b36bff", "hold": "#2ecc71"}
TRIAD_COLORS = ("#ff4040", "#40e040", "#4080ff")
OVERLAY_COLOR = "#ffe14d"
RESIDUAL_COLOR = "#ff2020"

HUD_FONT, P3_FONT, LINE_SPACING = 7.6, 6.8, 1.22
# figure fractions: [left, bottom, width, height]
RECT_3D = [0.005, 0.29, 0.36, 0.625]
RECT_HUD = [0.37, 0.255, 0.325, 0.69]
RECT_P3 = [0.70, 0.255, 0.295, 0.69]
RECT_TIMELINE = [0.06, 0.155, 0.90, 0.045]
RECT_SLIDER = [0.06, 0.112, 0.90, 0.025]


def _fmt(v, spec: str, none: str = "n/a") -> str:
    return none if v is None else format(v, spec)


class _QuietSlider(Slider):
    """A Slider that never draws by itself (its constructor would do a full
    draw): the viewer draws once per change, in _draw."""
    drawon = False


class _TextPanel(Artist):
    """Coloured monospace lines filling one axes, rasterised with Pillow (a
    matplotlib dependency) and drawn as one bitmap.

    Why not matplotlib Text: matplotlib 3.11's Agg backend lays out and
    renders text glyph by glyph in Python, about 40 us a glyph. These two
    panels hold about 5000 glyphs: 450 ms a frame that way, about 11 ms each
    through Pillow. Same font (matplotlib's monospace, DejaVu Sans Mono).
    Rendered at draw time at the axes' pixel size and the renderer's dpi
    (cached until the lines or the size change), so a resized window or a
    hi-dpi screen gets crisp text. The first line is drawn bold (the title).
    """

    def __init__(self, ax, fontsize: float, bg: str, pad_px: int = 6):
        super().__init__()
        self._ax, self.fontsize, self.bg, self.pad = ax, fontsize, bg, pad_px
        self.lines: List[Tuple[str, str]] = []
        self._fonts: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}
        self._key = None
        self._rgba: Optional[np.ndarray] = None
        self.rows = 0                   # rows drawn by the last render
        self.truncated = False          # lines left out for want of room
        self.set_zorder(1)
        ax.add_artist(self)

    def set_lines(self, lines: List[Tuple[str, str]]) -> None:
        self.lines = list(lines)
        self.stale = True

    def _font(self, px: int, bold: bool):
        if (px, bold) not in self._fonts:
            prop = font_manager.FontProperties(family="monospace",
                                               weight="bold" if bold else "normal")
            self._fonts[(px, bold)] = ImageFont.truetype(font_manager.findfont(prop), px)
        return self._fonts[(px, bold)]

    def _render(self, w: int, h: int, dpi: float) -> np.ndarray:
        px = max(6, int(round(self.fontsize * dpi / 72.0)))
        step = int(round(px * LINE_SPACING))
        width = max(20, int((w - 2 * self.pad) // self._font(px, False).getlength("M")))
        img = Image.new("RGBA", (w, h), self.bg)
        draw = ImageDraw.Draw(img)
        y, self.rows, self.truncated = self.pad, 0, False
        for k, (text, color) in enumerate(self.lines):
            for piece in textwrap.wrap(text, width, subsequent_indent="    ") or [""]:
                if y + step > h - self.pad:
                    self.truncated = True
                    break
                draw.text((self.pad, y), piece, font=self._font(px, k == 0), fill=color)
                y += step
                self.rows += 1
        return np.asarray(img)

    def draw(self, renderer) -> None:
        if not self.get_visible():
            return
        box = self._ax.get_window_extent(renderer)
        x0, y0 = int(round(box.x0)), int(round(box.y0))
        w, h = int(round(box.x1)) - x0, int(round(box.y1)) - y0
        if w < 2 or h < 2:
            return
        key = (tuple(self.lines), w, h, renderer.dpi)
        if key != self._key:
            self._rgba, self._key = self._render(w, h, renderer.dpi), key
        gc = renderer.new_gc()
        renderer.draw_image(gc, x0, y0, self._rgba[::-1])   # Agg: first row at the bottom
        gc.restore()
        self.stale = False


class ReplayViewer:
    """A `replay_core.Recording` in a matplotlib window (module docstring).

    State: frame_index (source frame of the displayed pose: i0), mode,
    reference, overlay, axes_on, playing, speed, displayed (the last
    DisplayedPose), media_t (its media time), shown (the arrays drawn: pos
    and rot in the current reference, recorded = the overlay's abs26 there,
    None when hidden). Methods that depend on time take an optional `now`
    (seconds on the clock given as `now=`, default time.perf_counter) so
    tests can drive them deterministically.
    """

    def __init__(self, rec: rc.Recording, *, mode: str = "hold",
                 reference: str = "camera", speed: float = 1.0,
                 start_frame: int = 0, overlay: bool = False,
                 axes: bool = False, figsize=(15.5, 8.6), dpi=None,
                 now=time.perf_counter,
                 screenshot_dir="results/replay_fidelity/screens",
                 timing_label: Optional[str] = None):
        if not rec.frames:
            raise ValueError("recording has no frames")
        if mode not in rc.MODES:
            raise ValueError(f"mode must be one of {rc.MODES}, got {mode!r}")
        if reference not in rc.REFERENCES:
            raise ValueError(f"reference must be one of {rc.REFERENCES}, "
                             f"got {reference!r}")
        self.rec = rec
        self.t = rec.timeline.t
        self.n = len(rec.frames)
        self.mode = mode
        self.reference = reference
        self.overlay = bool(overlay)
        self.axes_on = bool(axes)
        self._now = now
        self.clock = rc.PlaybackClock(speed=speed, now=now)
        self.interval_s = TICK_MS / 1000.0
        self.cls, self.cls_reason = p3.grasp_class(rec.name)
        self.hold = rc.hold_mask(rec)
        self._events_at: Dict[int, List[rc.Event]] = {}
        for e in rec.events:
            self._events_at.setdefault(e.index, []).append(e)
        self._recons = [rc.reconstruct(f) for f in rec.frames]
        self._limits = self._compute_limits()
        self._fid_cache: Dict[int, dict] = {}
        self._p3_cache: Dict[int, List[dict]] = {}
        # scheduling: one record per tick; skipped frames counted per play run
        self.sched = rc.SchedulingLog()
        self._tick_run: List[int] = []       # play run of each tick
        self._runs = 0
        self._last_shown: Optional[int] = None
        self._frames_skipped = 0
        self._next_intended: Optional[float] = None
        self._summary: Optional[dict] = None
        self._wall0 = float(now())
        self.speed_history = [{"wall_s": 0.0, "media_s": 0.0,
                               "speed": self.clock.speed}]
        backend = matplotlib.get_backend()
        self.timing_label = timing_label or (
            f"{'headless' if backend.lower() == 'agg' else 'interactive'} {backend}")
        self.screenshot_dir = Path(screenshot_dir)
        self._status = ""
        self._syncing = False
        self._keep: list = []               # single-shot timers of show()
        self.displayed: Optional[rc.DisplayedPose] = None
        self.media_t = 0.0
        self.frame_index = 0
        self._hud_lines: List[Tuple[str, str]] = []
        self._p3_lines: List[Tuple[str, str]] = []
        self._build_figure(figsize, dpi)
        self.seek_frame(start_frame)

    # --- state read through the clock ----------------------------------------
    @property
    def playing(self) -> bool:
        return self.clock.playing

    @property
    def speed(self) -> float:
        return self.clock.speed

    # --- precomputation --------------------------------------------------------
    def _compute_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Axis limits per reference, computed once from every reconstructed
        joint of the recording: camera the bounding box with a 10 % + 1 cm
        margin per axis, wrist/palm the same reach on both sides of the
        origin. The box aspect follows the ranges, so the scale is equal."""
        pts = np.concatenate([r.pos for r in self._recons])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = (hi - lo) * 0.1 + 0.01
        limits = {"camera": (lo - pad, hi + pad)}
        for ref in ("wrist", "palm"):
            local = np.concatenate([rc.to_reference(r.pos, r.rot, ref)[0]
                                    for r in self._recons])
            reach = np.abs(local).max(axis=0) * 1.08 + 0.01
            limits[ref] = (-reach, reach)
        return limits

    # --- figure ------------------------------------------------------------------
    def _build_figure(self, figsize, dpi) -> None:
        rec = self.rec
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=C_BG)
        take = f" take {rec.take}" if rec.take is not None else ""
        self.fig.suptitle(
            f"Replay: {rec.name}{take}  ({rec.kind}, {rec.hand_side} hand, "
            f"{self.n} frames)  skeleton only, as recorded",
            color="white", fontsize=11, y=0.988)
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(f"Replay {rec.name}{take}")
        handler = getattr(manager, "key_press_handler_id", None)
        if handler is not None:           # matplotlib's own keys (s, o, p, r, q, ...) clash
            self.fig.canvas.mpl_disconnect(handler)

        # 3D hand, with its frame label above and its notes below (figure
        # texts: kept out of the 3D axes so they are not redrawn with it)
        ax = self.ax = self.fig.add_axes(RECT_3D, projection="3d", facecolor=C_BG)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(colors="#202020", length=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.04, 0.04, 0.07, 1.0))
            axis.line.set_color("#1a1a1a")
        ax.set_autoscale_on(False)
        c = _HAND_COLORS.get(rec.hand_side, _HAND_COLORS["left"])
        zero2 = [[(0, 0, 0), (0, 0, 0)]]
        self._bones = Line3DCollection(zero2 * len(BONES), colors=c["bone"],
                                       linewidths=2.2)
        ax.add_collection3d(self._bones)
        self._palm = Poly3DCollection([[(0, 0, 0)] * len(PALM_POLYGON_IDX)],
                                      facecolors=c["palm"], edgecolors=c["bone"],
                                      linewidths=0.8, alpha=0.55)
        ax.add_collection3d(self._palm)
        self._joints = ax.scatter([0] * 26, [0] * 26, [0] * 26, c=c["joint"], s=22,
                                  edgecolors="none", depthshade=False)
        k = len(FINGERTIP_IDX)
        self._tips = ax.scatter([0] * k, [0] * k, [0] * k, c=c["tip"], s=90,
                                edgecolors="white", linewidths=0.6, depthshade=False)
        self._triads = []
        for color in TRIAD_COLORS:
            coll = Line3DCollection(zero2 * 26, colors=color, linewidths=1.2)
            ax.add_collection3d(coll)
            self._triads.append(coll)
        self._rec_markers = ax.scatter([0] * 26, [0] * 26, [0] * 26, s=70,
                                       facecolors="none", edgecolors=OVERLAY_COLOR,
                                       linewidths=1.0, depthshade=False)
        self._residuals = Line3DCollection(zero2 * 26, colors=RESIDUAL_COLOR,
                                           linewidths=1.5)
        ax.add_collection3d(self._residuals)
        top = RECT_3D[1] + RECT_3D[3]
        self._ref_text = self.fig.text(RECT_3D[0] + 0.005, top + 0.003, "",
                                       color=C_CYAN, family="monospace",
                                       fontsize=7.5, va="bottom", ha="left")
        self._note_3d = self.fig.text(RECT_3D[0] + 0.005, RECT_3D[1] - 0.003, "",
                                      color=C_DIM, family="monospace", fontsize=7.0,
                                      va="top", ha="left")
        w_in = self.fig.get_size_inches()[0]
        self._ref_width = int(RECT_3D[2] * w_in * 72 / (7.5 * 0.602)) - 2

        # panels: HUD (fidelity) and Paper 3 (research), visibly apart
        self.hud_ax = self._panel(RECT_HUD, "#2a2a35")
        self.p3_ax = self._panel(RECT_P3, C_P3_EDGE)
        self.hud_panel = _TextPanel(self.hud_ax, HUD_FONT, C_PANEL)
        self.p3_panel = _TextPanel(self.p3_ax, P3_FONT, C_P3_PANEL)

        # timeline markers above the slider, in source-frame-index coordinates
        self.tl_ax = self.fig.add_axes(RECT_TIMELINE, facecolor="#101018")
        self._build_timeline()

        self.slider_ax = self.fig.add_axes(RECT_SLIDER, facecolor="#202028")
        self.slider = _QuietSlider(self.slider_ax, "frame", 0, max(1, self.n - 1),
                                   valinit=0, valstep=1, valfmt="%d",
                                   color="#22e0e0")
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        specs = [("play", "Play", self.toggle_play),
                 ("slower", "Speed -", lambda: self.cycle_speed(-1)),
                 ("faster", "Speed +", lambda: self.cycle_speed(+1)),
                 ("back", "Step -1", lambda: self.step(-1)),
                 ("fwd", "Step +1", lambda: self.step(+1)),
                 ("prev", "Prev event", self.prev_event),
                 ("next", "Next event", self.next_event),
                 ("ref", "Ref", self.cycle_reference),
                 ("mode", "Mode", self.toggle_mode),
                 ("overlay", "Overlay", self.toggle_overlay),
                 ("axes", "Axes", self.toggle_axes),
                 ("shot", "Screenshot", self._screenshot_from_gui)]
        self.buttons: Dict[str, Button] = {}
        for k, (key, label, fn) in enumerate(specs):
            bax = self.fig.add_axes([0.06 + k * 0.075, 0.04, 0.07, 0.045])
            btn = Button(bax, label, color="#2a2a35", hovercolor="#3a3a48")
            btn.label.set_color("white")
            btn.label.set_fontsize(8)
            btn.on_clicked(lambda _e, fn=fn: fn())
            self.buttons[key] = btn
        self.fig.text(0.06, 0.012,
                      "keys: space play/pause   +/- speed   left/right step   "
                      "p/n prev/next event   r reference   i hold/interpolate   "
                      "o overlay   a axes   s screenshot   q/esc quit",
                      color=C_DIM, family="monospace", fontsize=7.5)

        # Blitting: what changes from frame to frame is animated, so a full
        # draw leaves it out; the draw_event handler keeps that full draw as
        # the background and draws the changing parts on top. A frame then
        # restores the background and redraws only those (_draw).
        self._dynamic = [self.ax, self.hud_ax, self.p3_ax, self.slider_ax,
                         self._cursor]
        for a in self._dynamic:
            a.set_animated(True)
        self._bg = None
        self._bg_key = None
        self.fig.canvas.mpl_connect("draw_event", self._on_draw_event)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._timer = self.fig.canvas.new_timer(interval=TICK_MS)
        self._timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("close_event", lambda _e: self._timer.stop())
        self._apply_reference()

    def _panel(self, rect, edge):
        pax = self.fig.add_axes(rect)
        pax.set_xticks([]); pax.set_yticks([])
        for spine in pax.spines.values():
            spine.set_color(edge)
        return pax

    def _build_timeline(self) -> None:
        """One marker per event (kind, index, artists), plus one for the hold
        band when the metadata declares a static interval."""
        ax, n = self.tl_ax, self.n
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=C_DIM, labelsize=7, length=2)
        for spine in ax.spines.values():
            spine.set_color("#2a2a35")
        self.timeline_markers: List[Tuple[str, Optional[int], list]] = []
        if self.hold is not None and self.hold.any():
            idx = np.flatnonzero(self.hold)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            arts = [ax.axvspan(r[0] - 0.5, r[-1] + 0.5, color=EVENT_COLORS["hold"],
                               alpha=0.30, lw=0, zorder=0) for r in runs]
            self.timeline_markers.append(("hold", None, arts))
        for e in self.rec.events:
            i, color = e.index, EVENT_COLORS.get(e.kind, "white")
            if e.kind == "gap":           # between frame i-1 and i
                arts = [ax.axvspan(i - 1, i, color=color, alpha=0.85, lw=0, zorder=2),
                        ax.axvline(i - 0.5, color=color, lw=3, zorder=2)]
            elif e.kind == "id_change":
                arts = [ax.axvline(i - 0.5, ymin=0.45, ymax=1.0, color=color,
                                   lw=2.5, zorder=3),
                        ax.plot([i - 0.5], [0.9], marker="v", color=color,
                                ms=7, zorder=3)[0]]
            elif e.kind == "dropped_frames":
                arts = [ax.axvline(i - 0.5, ymin=0.0, ymax=0.4, color=color,
                                   lw=1.5, zorder=3)]
            elif e.kind == "invalid_status":
                arts = [ax.axvline(i, ymin=0.3, ymax=0.7, color=color, lw=1.5,
                                   zorder=3)]
            else:                         # clock_defect
                arts = [ax.plot([i], [0.5], marker="D", color=color, ms=6,
                                zorder=4)[0]]
            self.timeline_markers.append((e.kind, i, arts))
        self._cursor = ax.axvline(0, color="white", lw=1.2, zorder=5)
        handles = [Patch(color=EVENT_COLORS["gap"], label="gap"),
                   Line2D([], [], color=EVENT_COLORS["id_change"], lw=2.5,
                          marker="v", label="id_change"),
                   Line2D([], [], color=EVENT_COLORS["dropped_frames"], lw=1.5,
                          label="dropped_frames"),
                   Line2D([], [], color=EVENT_COLORS["invalid_status"], lw=1.5,
                          label="invalid_status"),
                   Line2D([], [], color=EVENT_COLORS["clock_defect"], lw=0,
                          marker="D", label="clock_defect"),
                   Patch(color=EVENT_COLORS["hold"], alpha=0.5,
                         label="hold (static interval)"),
                   Line2D([], [], color="white", lw=1.2, label="current frame")]
        leg = ax.legend(handles=handles, ncol=len(handles), loc="lower left",
                        bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=7.5,
                        handlelength=1.6, columnspacing=1.4, borderaxespad=0.2)
        for text in leg.get_texts():
            text.set_color(C_TEXT)
        ax.text(1.0, 1.08, f"timeline: source frame index 0..{n - 1}, "
                f"{len(self.rec.events)} events", transform=ax.transAxes,
                ha="right", va="bottom", color=C_DIM, fontsize=7.5)

    # --- what is shown -------------------------------------------------------------
    def _render_at(self, t: float) -> None:
        self.displayed = rc.pose_at(self.rec, float(t), self.mode)
        self.media_t = float(t)
        if self.displayed.i0 is not None:
            self.frame_index = self.displayed.i0
        self._draw()

    def _draw(self) -> None:
        self._draw_hand()
        self._hud_lines = self._compose_hud()
        self._status = ""                   # a status line is shown by one draw only
        self.hud_panel.set_lines(self._hud_lines)
        self._p3_lines = self._compose_paper3()
        self.p3_panel.set_lines(self._p3_lines)
        self._cursor.set_xdata([self.frame_index, self.frame_index])
        self._sync_widgets()
        canvas = self.fig.canvas
        if self._bg is not None and self._bg_key == self._static_key():
            canvas.restore_region(self._bg)
            renderer = canvas.get_renderer()
            for a in self._dynamic:
                a.draw(renderer)
            canvas.blit(self.fig.bbox)
        else:                               # something static changed: full draw
            canvas.draw_idle()

    def _static_key(self) -> tuple:
        """Everything outside the animated artists that can change; a
        background is reused only while this is unchanged."""
        return (self._ref_text.get_text(), self._note_3d.get_text(),
                tuple(b.label.get_text() for b in self.buttons.values()),
                tuple(self.fig.canvas.get_width_height()), self.fig.dpi)

    def _on_draw_event(self, event) -> None:
        """After every full draw (window, resize, savefig): keep it as the
        background, unless it is a savefig or a layout pass, then draw the
        animated artists on top with the same renderer."""
        canvas = self.fig.canvas
        if (not canvas.is_saving() and hasattr(canvas, "copy_from_bbox")
                and getattr(canvas, "renderer", None) is event.renderer):
            self._bg = canvas.copy_from_bbox(self.fig.bbox)
            self._bg_key = self._static_key()
        for a in self._dynamic:
            a.draw(event.renderer)

    def _recorded_in_reference(self, d: rc.DisplayedPose) -> np.ndarray:
        """abs26 moved by the same transform as the reconstruction, so the
        residual is the same in every reference."""
        rec_abs = d.frame.abs_recorded
        if self.reference == "camera":
            return rec_abs
        axes = d.rot[rc.WRIST] if self.reference == "wrist" else d.rot[rc.PALM]
        return (rec_abs - d.pos[rc.WRIST]) @ axes

    def _draw_hand(self) -> None:
        d = self.displayed
        hand = [self._bones, self._palm, self._joints, self._tips]
        self.shown = {"pos": None, "rot": None, "recorded": None}
        if d is None or d.pos is None:
            for a in hand + self._triads + [self._rec_markers, self._residuals]:
                a.set_visible(False)
            self._note_3d.set_text("nothing to show: media time before the first frame")
            return
        pos, rot = rc.to_reference(d.pos, d.rot, self.reference)
        self.shown.update(pos=pos, rot=rot)
        self._bones.set_segments([[pos[a], pos[b]] for a, b in BONES])
        self._palm.set_verts([pos[PALM_POLYGON_IDX]])
        self._joints._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        tips = pos[FINGERTIP_IDX]
        self._tips._offsets3d = (tips[:, 0], tips[:, 1], tips[:, 2])
        for a in hand:
            a.set_visible(True)
        for k, coll in enumerate(self._triads):
            if self.axes_on:
                coll.set_segments([[pos[j], pos[j] + TRIAD_M * rot[j][:, k]]
                                   for j in range(len(pos))])
            coll.set_visible(self.axes_on)
        show = (self.overlay and d.source in ("recorded", "held")
                and d.frame.abs_recorded is not None)
        notes = []
        if show:
            recp = self._recorded_in_reference(d)
            self.shown["recorded"] = recp
            self._rec_markers._offsets3d = (recp[:, 0], recp[:, 1], recp[:, 2])
            self._residuals.set_segments([[pos[j], recp[j]] for j in range(len(pos))])
            notes.append("overlay: recorded abs26 = hollow markers; residual "
                         "segments red, to scale (values in the HUD)")
        elif self.overlay:
            notes.append(RECORDED_NA if d.source == "interpolated"
                         else "recorded: n/a (this line has no abs26)")
        self._rec_markers.set_visible(show)
        self._residuals.set_visible(show)
        if self.axes_on:
            notes.append("joint axes: 1 cm, x red, y green, z blue")
        lo, hi = self._limits[self.reference]
        size = hi - lo
        notes.append(f"view box x {size[0]:.3f}  y {size[1]:.3f}  z {size[2]:.3f} m "
                     f"(equal scale, set once per reference)")
        self._note_3d.set_text("\n".join(notes))

    def _segment_of(self, i: int) -> Tuple[int, rc.Segment]:
        for k, s in enumerate(self.rec.segments, start=1):
            if s.start <= i <= s.end:
                return k, s
        raise ValueError(f"frame index {i} in no segment")

    def _fidelity(self, i: int) -> dict:
        if i not in self._fid_cache:
            self._fid_cache[i] = rc.frame_fidelity(self.rec.frames[i])
        return self._fid_cache[i]

    def _paper3_rows(self) -> List[dict]:
        d = self.displayed
        if d.source in ("recorded", "held"):
            if d.i0 not in self._p3_cache:
                r = self._recons[d.i0]
                self._p3_cache[d.i0] = p3.frame_relation_values(
                    p3.frame_angles(r.pos, r.rot, self.rec.hand_side), self.cls)
            return self._p3_cache[d.i0]
        return p3.frame_relation_values(
            p3.frame_angles(d.pos, d.rot, self.rec.hand_side), self.cls)

    def _pose_line(self) -> str:
        d = self.displayed
        if d.source == "recorded":
            return "displayed pose: recorded"
        if d.source == "interpolated":
            extra = f"; {d.note}" if d.note else ""
            return (f"displayed pose: interpolated ({d.i0}..{d.i1} alpha "
                    f"{d.alpha:.3f}){extra}")
        if d.source == "held":
            return ("displayed pose: held ("
                    + (d.note or "latest source frame at or before the media time")
                    + ")")
        return f"displayed pose: unavailable ({d.note})"

    def _compose_hud(self) -> List[Tuple[str, str]]:
        rec, d, tl = self.rec, self.displayed, self.rec.timeline
        ids = " ".join(str(h) for h in rec.hand_ids) if rec.hand_ids else "any"
        take = rec.take if rec.take is not None else "-"
        ex = rec.lines_excluded
        out = [("Replay HUD (fidelity = this replay vs the recording itself)", C_TITLE),
               (f"recording: {rec.path.name}", C_TEXT),
               (f"kind {rec.kind}   name {rec.name}   take {take}   hand "
                f"{rec.hand_side}   hand ids {ids}", C_TEXT),
               (f"provenance: {rec.provenance or '(none given)'}", C_DIM),
               (f"lines: {rec.lines_total} in the file, {self.n} selected; excluded "
                f"other_hand {ex.get('other_hand', 0)}, other_id "
                f"{ex.get('other_id', 0)}, invalid_status "
                f"{ex.get('invalid_status', 0)}", C_DIM),
               ("", C_TEXT)]
        f = d.frame
        if f is None:
            out.append(("frame: none (media time before the first frame)", C_WARN))
        else:
            k, seg = self._segment_of(f.index)
            where = (f"frames {d.i0}..{d.i1} of {self.n}" if d.source == "interpolated"
                     else f"frame {f.index}/{self.n}")
            out.append((f"{where}   frame_id {f.frame_id}   hand_id {f.hand_id}   "
                        f"status {f.status}   segment {k}/{len(rec.segments)} "
                        f"(hand_id {seg.hand_id}, frames {seg.start}-{seg.end})",
                        C_CYAN))
            tags = []
            if self.hold is not None and self.hold[f.index]:
                tags.append("inside the static interval (hold)")
            if str(f.frame_id) == str(rec.metadata.get("representative_frame_id")):
                tags.append("the metadata's representative frame")
            if tags:
                out.append(("  " + "; ".join(tags), C_GOOD))
        out.append((f"media time {self.media_t:.6f} s   (recording 0 .. "
                    f"{float(self.t[-1]):.6f} s on {tl.source})", C_CYAN))
        if tl.fallback:
            out.append((f"clock: {tl.source} (FALLBACK, not timestamp_us): "
                        f"{tl.reason}", C_WARN))
        else:
            out.append((CLOCK_OK, C_GOOD))
        if f is not None:
            out.append((f"stamps of frame_id {f.frame_id}: wall_time {f.wall_time!r}"
                        f"   capture_time {f.capture_time!r}   timestamp_us "
                        f"{f.timestamp_us}", C_TEXT))
            out.extend(self._delta_lines(f))
        out.append((self._pose_line(), C_TEXT if d.source == "recorded" else C_WARN))
        if f is not None:
            for e in self._events_at.get(f.index, []):
                out.append((f"event at this frame: {e.kind}: {e.detail}", C_WARN))
        out.append(("", C_TEXT))
        out.extend(self._fidelity_lines())
        if self.overlay:
            if d.source == "interpolated":
                out.append((RECORDED_NA, C_WARN))
            elif f is not None and f.abs_recorded is None:
                out.append(("recorded: n/a (this line has no abs26)", C_WARN))
            else:
                out.append(("recorded: abs26 as hollow markers, residual segments "
                            "red (to scale)", C_DIM))
        out.append(("units: positions m (camera space), residuals µm, angles deg, "
                    "time s", C_DIM))
        out.append((f"reference: {REFERENCE_LABELS[self.reference]}", C_CYAN))
        out.append(("", C_TEXT))
        out.append((f"speed {self.speed:g}x   mode {self.mode}   overlay "
                    f"{'on' if self.overlay else 'off'}   axes "
                    f"{'on' if self.axes_on else 'off'}   "
                    f"{'PLAYING' if self.playing else 'paused'}", C_TEXT))
        s = self.timing_summary()
        if s["n"]:
            out.append((f"scheduling ({s['n']} ticks; lateness vs timer slot, media "
                        f"ms): mean {s['mean_ms']:.1f}  p95 {s['p95_ms']:.1f}  max "
                        f"{s['max_ms']:.1f}  frames skipped {s['frames_skipped']}",
                        C_DIM))
        else:
            out.append(("scheduling: no ticks logged yet (logged while playing)", C_DIM))
        if self._status:
            out.append((self._status, C_TITLE))
        return out

    def _delta_lines(self, f: rc.SourceFrame) -> List[Tuple[str, str]]:
        if f.index == 0:
            return [("first selected frame: no previous frame", C_DIM)]
        prev = self.rec.frames[f.index - 1]

        def diff(a, b, scale):
            return None if a is None or b is None else (a - b) * scale
        dw = diff(f.wall_time, prev.wall_time, 1e3)
        dc = diff(f.capture_time, prev.capture_time, 1e3)
        dt = diff(f.timestamp_us, prev.timestamp_us, 1)
        out = [(f"since frame_id {prev.frame_id}: wall_time {_fmt(dw, '+.3f')} ms   "
                f"capture_time {_fmt(dc, '+.3f')} ms   timestamp_us "
                f"{_fmt(dt, '+d')} µs", C_DIM)]
        if dw is not None and dt is not None and dw / 1e3 - dt / 1e6 > WRITER_STALL_S:
            out.append((f"writer stall: wall_time advanced {dw:.1f} ms while the "
                        f"camera clock advanced {dt / 1e3:.3f} ms; playback follows "
                        f"the camera clock", C_WARN))
        return out

    def _fidelity_lines(self) -> List[Tuple[str, str]]:
        d = self.displayed
        out = [("fidelity of the displayed source frame (replay vs recording):",
                C_TITLE)]
        if d.source not in ("recorded", "held"):
            out.append(("  n/a: the pose is interpolated between source frames and "
                        "has no recorded counterpart", C_DIM))
            return out
        ff = self._fidelity(d.i0)
        tol = rc.TOLERANCES
        pr = ff["position_residual_m"]
        if pr["max"] is None:
            out.append((f"  position residual: n/a (this line has no abs26) "
                        f"[{FID_POSITION}]", C_WARN))
        else:
            ok = pr["max"] <= tol["position_m"]["value"]
            out.append((f"  position residual max {pr['max'] * 1e6:.3f} µm, mean "
                        f"{pr['mean'] * 1e6:.3f} µm (tolerance "
                        f"{tol['position_m']['value'] * 1e6:g} µm): {FID_POSITION}",
                        C_TEXT if ok else C_BAD))
        cd, cm = ff["chain_deg"]["max"], ff["chain_m"]["max"]
        ok = cd <= tol["chain_deg"]["value"] and cm <= tol["chain_m"]["value"]
        out.append((f"  chain cross-check max {cd:.2e} deg, {cm:.2e} m (tolerance "
                    f"{tol['chain_deg']['value']:g} deg, {tol['chain_m']['value']:g} "
                    f"m): {FID_CHAIN}", C_TEXT if ok else C_BAD))
        bd = ff["bone_direction_deg"]["max"]
        exp = tol["bone_direction_deg_expectation"]["value"]
        out.append((f"  bone-direction max {_fmt(bd, '.4f')} deg (expectation "
                    f"{exp:g} deg, reported): {FID_BONE}",
                    C_TEXT if bd is None or bd <= exp else C_WARN))
        return out

    def _compose_paper3(self) -> List[Tuple[str, str]]:
        d = self.displayed
        out = [(P3_TITLE, C_P3_TITLE),
               (f"class: {self.cls} ({self.cls_reason})", C_TEXT)]
        if d.pos is None:
            out.append(("no pose at this media time", C_WARN))
            return out
        if d.source == "interpolated":
            a, b = self.rec.frames[d.i0], self.rec.frames[d.i1]
            out.append((f"angles of the displayed pose: interpolated between "
                        f"frame_id {a.frame_id} and {b.frame_id} (not a recorded "
                        f"pose)", C_WARN))
        else:
            out.append((f"angles of source frame_id {d.frame.frame_id}", C_DIM))
        out.append(("Cobos et al. 2009; residual = lhs - coeff × rhs, degrees; no "
                    "pass/fail threshold (the paper gives none)", C_DIM))
        out.append((f"{'relation':<22} {'status':<13} {'lhs':>7} "
                    f"{'coeff × rhs':>19} {'resid':>7}", C_CYAN))
        for r in self._paper3_rows():
            if r["residual"] == "unavailable":
                line = f"{r['id']:<22} {r['status']:<13} unavailable"
            else:
                prod = float(Fraction(r["coeff"])) * r["rhs_value"]
                line = (f"{r['id']:<22} {r['status']:<13} {r['lhs_value']:7.1f} "
                        f"{r['coeff']:>5}×{r['rhs_value']:6.1f}={prod:6.1f} "
                        f"{r['residual']:+7.1f}")
                if r["kind"] == "bound":
                    line += (f" ≤{r['bound_deg']:g}:"
                             f"{'yes' if r['satisfied'] else 'no'}")
            out.append((line, STATUS_COLORS.get(r["status"], C_TEXT)))
        out.append(("", C_TEXT))
        first = p3.COMPARE_NOTE.split(". ")[0].rstrip(".") + "."
        out.append((f"MCP order: {first}", C_DIM))
        out.append(("unavailable (not given by the camera): "
                    + ", ".join(p3.UNAVAILABLE), C_DIM))
        return out

    def _sync_widgets(self) -> None:
        self._syncing = True
        try:
            if int(self.slider.val) != self.frame_index:
                self.slider.set_val(self.frame_index)
        finally:
            self._syncing = False
        b = self.buttons
        b["play"].label.set_text("Pause" if self.playing else "Play")
        b["ref"].label.set_text(f"Ref: {self.reference}")
        b["mode"].label.set_text(f"Mode: {self.mode}")
        b["overlay"].label.set_text(f"Overlay: {'on' if self.overlay else 'off'}")
        b["axes"].label.set_text(f"Axes: {'on' if self.axes_on else 'off'}")

    def _apply_reference(self) -> None:
        """Limits, box aspect, first view and labels of the reference; the
        view resets here and nowhere else (never per frame)."""
        lo, hi = self._limits[self.reference]
        elev, azim = VIEWS[self.reference]
        self.ax.view_init(elev=elev, azim=azim, vertical_axis="y")
        self.ax.set_xlim(lo[0], hi[0])
        self.ax.set_ylim(lo[1], hi[1])
        self.ax.set_zlim(lo[2], hi[2])
        self.ax.set_box_aspect(tuple(hi - lo))
        text = (f"{REFERENCE_LABELS[self.reference]}; "
                f"{PLOT_MAPPING[self.reference]}")
        self._ref_text.set_text("\n".join(textwrap.wrap(text, self._ref_width)))
        names = (("x (right)", "y (up)", "z (toward operator)")
                 if self.reference == "camera" else ("x", "y (dorsal)", "z (proximal)"))
        for setter, name in zip((self.ax.set_xlabel, self.ax.set_ylabel,
                                 self.ax.set_zlabel), names):
            setter(name, color=C_DIM, fontsize=8, labelpad=-8)

    # --- controls ----------------------------------------------------------------
    def _new_run(self, now=None) -> None:
        """A play run starts (play, or a seek or speed change while playing)
        from the frame on screen; the next tick is due one interval later."""
        self._runs += 1
        self._last_shown = self.frame_index
        self._next_intended = (self.clock.media_time(now)
                               + self.interval_s * self.clock.speed)

    def _pause_clock(self, now) -> float:
        """Pause, clamped to the end of the recording; the media time."""
        self.clock.pause(now)
        self._timer.stop()
        end = float(self.t[-1])
        if self.clock.media_time(now) > end:
            self.clock.seek(end, now)
        return self.clock.media_time(now)

    def toggle_play(self, now=None) -> None:
        n = float(self._now() if now is None else now)
        if self.playing:
            self._status = "paused"
            self._render_at(self._pause_clock(n))         # what is shown = the clock
            return
        if self.media_t >= float(self.t[-1]):             # at the end: from the start
            self.clock.seek(float(self.t[0]), n)
        self.clock.start(n)
        self._timer.start()
        self._render_at(self.clock.media_time(n))
        self._new_run(n)

    def set_speed(self, x: float, now=None) -> None:
        n = float(self._now() if now is None else now)
        self.clock.set_speed(x, n)
        self.speed_history.append({"wall_s": n - self._wall0,
                                   "media_s": self.clock.media_time(n),
                                   "speed": self.clock.speed})
        if self.playing:
            self._new_run(n)
        self._draw()

    def cycle_speed(self, direction: int, now=None) -> None:
        """Next listed speed up (direction > 0) or down; stays at the ends."""
        s = self.speed
        if direction > 0:
            faster = [v for v in SPEEDS if v > s + 1e-12]
            target = faster[0] if faster else SPEEDS[-1]
        else:
            slower = [v for v in SPEEDS if v < s - 1e-12]
            target = slower[-1] if slower else SPEEDS[0]
        self.set_speed(target, now)

    def seek_time(self, t: float, now=None) -> None:
        """Show media time t (clamped to the recording); keeps playing if it was."""
        t = float(min(max(float(t), float(self.t[0])), float(self.t[-1])))
        self.clock.seek(t, now)
        self._render_at(t)
        if self.playing:
            self._new_run(now)

    def seek_frame(self, i: int, now=None) -> None:
        """Show source frame i at its exact time (clamped to 0..n-1)."""
        i = int(min(max(int(i), 0), self.n - 1))
        self.seek_time(float(self.t[i]), now)

    def _pause_for_inspection(self, now=None) -> None:
        if self.playing:
            self._pause_clock(now)

    def step(self, delta: int, now=None) -> None:
        """Pause and move by `delta` source frames; from a pose between two
        frames, -1 lands on the earlier one."""
        self._pause_for_inspection(now)
        base = self.frame_index
        if delta < 0 and self.media_t > float(self.t[base]):
            base += 1
        self.seek_frame(base + int(delta), now)

    def _event_indices(self) -> List[int]:
        return sorted(self._events_at)

    def _jump_to_event(self, targets: List[int], none_text: str, now=None) -> None:
        self._pause_for_inspection(now)
        if not targets:
            self._status = none_text
            self._draw()
            return
        kinds = ", ".join(e.kind for e in self._events_at[targets[0]])
        self._status = f"jumped to the event at frame {targets[0]}: {kinds}"
        self.seek_frame(targets[0], now)

    def next_event(self, now=None) -> None:
        """Pause and jump to the next frame carrying an event (rec.events)."""
        self._jump_to_event([i for i in self._event_indices() if i > self.frame_index],
                            "no later event", now)

    def prev_event(self, now=None) -> None:
        """Pause and jump to the previous frame carrying an event; from a pose
        held between two frames, an event on the held frame counts as earlier."""
        between = self.media_t > float(self.t[self.frame_index])
        earlier = [i for i in self._event_indices()
                   if i < self.frame_index or (between and i == self.frame_index)]
        self._jump_to_event(earlier[::-1], "no earlier event", now)

    def set_reference(self, name: str) -> None:
        if name not in rc.REFERENCES:
            raise ValueError(f"reference must be one of {rc.REFERENCES}, got {name!r}")
        self.reference = name
        self._apply_reference()
        self._draw()

    def cycle_reference(self) -> None:
        refs = rc.REFERENCES
        self.set_reference(refs[(refs.index(self.reference) + 1) % len(refs)])

    def set_mode(self, name: str) -> None:
        if name not in rc.MODES:
            raise ValueError(f"mode must be one of {rc.MODES}, got {name!r}")
        self.mode = name
        self._render_at(self.media_t)

    def toggle_mode(self) -> None:
        self.set_mode("interpolate" if self.mode == "hold" else "hold")

    def toggle_overlay(self) -> None:
        self.overlay = not self.overlay
        self._draw()

    def toggle_axes(self) -> None:
        self.axes_on = not self.axes_on
        self._draw()

    # --- playback ----------------------------------------------------------------
    def tick(self, now=None) -> Optional[rc.DisplayedPose]:
        """One timer tick (module docstring, TIMING_DOC). Does nothing while
        paused; at the end of the recording shows the last frame and pauses."""
        if not self.playing:
            return self.displayed
        t_now = float(self._now() if now is None else now)
        reached = self.clock.media_time(t_now)
        media = reached
        if media >= float(self.t[-1]):
            media = self._pause_clock(t_now)
            self._status = "end of the recording: paused"
        self.displayed = rc.pose_at(self.rec, media, self.mode)
        self.media_t = media
        if self.displayed.i0 is not None:
            self.frame_index = self.displayed.i0
        self.sched.record(self._next_intended, reached, self.frame_index)
        self._tick_run.append(self._runs)
        if self._last_shown is not None:
            self._frames_skipped += max(0, self.frame_index - self._last_shown - 1)
        self._last_shown = self.frame_index
        self._next_intended = reached + self.interval_s * self.clock.speed
        self._summary = None
        self._draw()
        return self.displayed

    def timing_summary(self) -> dict:
        """SchedulingLog.stats() over every tick (n, mean_ms, p95_ms, max_ms),
        frames_skipped counted inside play runs (the log's own count would
        pair the last tick before a seek with the first one after it), plus
        what is logged, the speed history and the backend."""
        if self._summary is None:
            self._summary = {**self.sched.stats(),
                             "frames_skipped": self._frames_skipped}
        return {**self._summary, "runs": self._runs,
                "speed_history": [dict(h) for h in self.speed_history],
                "backend": matplotlib.get_backend(), "label": self.timing_label,
                "timer_interval_ms": TICK_MS, "what_is_logged": TIMING_DOC}

    def write_timing_log(self, path, label: Optional[str] = None) -> Path:
        """timing_summary() plus every tick, as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = self.timing_summary()
        if label:
            out["label"] = label
        rec = self.rec
        out["recording"] = {"path": str(rec.path), "kind": rec.kind,
                            "name": rec.name, "take": rec.take,
                            "frames": self.n, "clock": rec.timeline.source}
        out["written"] = datetime.now().isoformat(timespec="seconds")
        s = self.sched
        out["ticks"] = [{"run": k, "intended_media_s": a, "actual_media_s": b,
                         "lateness_ms": abs(b - a) * 1e3, "frame_index": i}
                        for k, a, b, i in zip(self._tick_run, s.intended, s.actual,
                                              s.frame_index)]
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        return path

    # --- output ------------------------------------------------------------------
    def hud_text(self) -> str:
        """The HUD as shown (unwrapped lines)."""
        return "\n".join(text for text, _ in self._hud_lines)

    def paper3_text(self) -> str:
        """The Paper 3 panel as shown (unwrapped lines)."""
        return "\n".join(text for text, _ in self._p3_lines)

    def screenshot(self, path=None) -> Path:
        """The whole figure as a PNG; without a path, a stamped file in
        screenshot_dir."""
        if path is None:
            take = f"_take{self.rec.take}" if self.rec.take is not None else ""
            fid = self.displayed.frame.frame_id if self.displayed.frame else "none"
            path = self.screenshot_dir / (
                f"{self.rec.name}{take}_frame{fid}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, facecolor=self.fig.get_facecolor())
        return path

    def _screenshot_from_gui(self) -> None:
        path = self.screenshot()
        print(f"screenshot saved: {path}")
        self._status = f"screenshot saved: {path}"
        self._draw()

    def screenshot_timeline(self, path) -> Path:
        """The timeline with its legend and the slider, cropped."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        box = Bbox.union([self.tl_ax.get_tightbbox(renderer),
                          self.slider_ax.get_tightbbox(renderer)])
        inches = box.transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(path, facecolor=self.fig.get_facecolor(),
                         bbox_inches=inches.padded(0.08))
        return path

    # --- GUI -----------------------------------------------------------------------
    def _on_slider(self, val) -> None:
        if not self._syncing:
            self.seek_frame(int(round(val)))

    def _on_key(self, event) -> None:
        actions = {
            " ": self.toggle_play, "space": self.toggle_play,
            "+": lambda: self.cycle_speed(+1), "=": lambda: self.cycle_speed(+1),
            "-": lambda: self.cycle_speed(-1), "_": lambda: self.cycle_speed(-1),
            "left": lambda: self.step(-1), "right": lambda: self.step(+1),
            "p": self.prev_event, "n": self.next_event,
            "r": self.cycle_reference, "i": self.toggle_mode,
            "o": self.toggle_overlay, "a": self.toggle_axes,
            "s": self._screenshot_from_gui, "q": self.close, "escape": self.close,
        }
        if event.key in actions:
            actions[event.key]()

    def show(self, autoplay: bool = False, duration: float = 0.0) -> None:
        """Run the GUI loop. autoplay starts playing once the window is up;
        duration (s) closes the window that long after playback starts."""
        if autoplay or duration:
            start = self.fig.canvas.new_timer(interval=200)
            start.single_shot = True
            start.add_callback(lambda: None if self.playing else self.toggle_play())
            start.start()
            self._keep.append(start)
        if duration:
            stop = self.fig.canvas.new_timer(interval=int((duration + 0.2) * 1000))
            stop.single_shot = True
            stop.add_callback(self.close)
            stop.start()
            self._keep.append(stop)
        plt.show()

    def close(self) -> None:
        self._timer.stop()
        plt.close(self.fig)
