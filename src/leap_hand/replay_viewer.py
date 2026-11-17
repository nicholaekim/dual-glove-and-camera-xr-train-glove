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

THREE PANELS, KEPT APART
  HUD        replay fidelity: what is shown, from which line, on which clock,
             and how far the reconstruction is from the recording itself.
  Angles     every angle of the displayed pose the coverage table names
             (`angle_coverage.all_angles`), grouped by digit, each with its
             column name and status (calculated, outside Paper 3); the finger
             MCP pairs twice, flexion-first (xyz) and the paper's order; the
             twist of each bone on its digit's line; the six coordinates the
             camera does not measure named, never filled (the title says
             what "calculated" is: a tracker coordinate); with an optional
             `mapping_provider`, the two-row "reference-mapped" thumb TMC
             block (leap_hand.reference_validation: a mapping evaluated
             against an independent reference, its values only inside its
             scope, its tag saying reference-evaluated, SYNTHETIC or
             cross-check; nothing without one); and apart, the
             "estimated" thumb TMC block, fed by an optional `tmc_provider`
             (not computed without one): the two estimate values named by
             the calibration's kind (`_est_openhand` for the session's open
             hand, `_est_neutral` for a tmc_neutral calibration take; the
             block's title says which), the twist about the bone with its
             residual when the provider gives them, and the calibration's
             provenance. The script scripts/leap/replay_view.py builds the
             provider from a calibration.json of either kind
             (`leap_hand.tmc_estimate`, `leap_hand.tmc_calibration_take`).
  Paper 3    the research comparison (`paper3.frame_relation_values`), with
             the class, the MCP-order disclosure and the six angles the
             camera does not give. Not a fidelity number.
  All three are monospace text rasterised with Pillow (`_TextPanel`, why there).

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
  (`write_timing_log`) carry the statistics, with the rendered updates, the
  distinct frames shown and the wall duration of each play run counted
  apart from the ticks. Playback stops at the end.

RENDERING (profiled 2026-10-24 on the TkAgg window, docs/replay_fidelity.md
section 7)
  A full draw paints everything static (the 3D panes, grid, ticks and labels,
  the panel frames, the timeline markers, the buttons) and is kept as the
  background. A tick restores it and redraws only what moves: the hand's 3D
  artists (projected with the axes' current matrix, far to near), the three
  text panels, the slider and the timeline cursor, then blits the four
  regions they occupy. The text panels compose their bitmap from a glyph
  atlas (`_TextPanel`). Before this the whole 3D axes was redrawn each tick
  (14 ms, nearly all of it the static decorations), each panel was rendered
  line by line through Pillow (30 ms for the three) and the whole figure was
  blitted (10 ms): 64 ms a tick, about 11 updates a second at 1x.
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
from PIL import Image, ImageColor, ImageDraw, ImageFont

from xr_hand.joints import BONES
from xr_hand.viz3d import _HAND_COLORS, FINGERTIP_IDX, PALM_POLYGON_IDX

from . import angle_coverage as ac
from . import paper3 as p3
from . import replay_core as rc
from . import tmc_estimate as te

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
# Tk's after() on Windows rounds every interval above 1 ms up to the 15.6 ms
# system timer quantum (measured 2026-10-24: after(5) and after(10) both fire
# every 15.6 ms, after(1) every 1.0 ms), so with a 15 ms interval a tick came
# 15.6 to 31 ms after the previous draw ended. With 1 ms the next tick follows
# the draw at once and the draw time alone sets the rate; the media clock is
# wall time, so the interval changes nothing about which frame is shown.
TICK_MS = 1
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
ANG_TITLE = "Angles of the displayed source frame (degrees; calculated = tracker coordinate)"
ANG_SIGNS = "flexion + toward the palm; abduction + toward the thumb side; both hands"
ANG_TWIST = ("digit lines: twist of each bone, <DIGIT>_<BONE>_twist_tracker "
             "(outside Paper 3)")
ANG_UNAVAILABLE = ("unavailable: " + ", ".join(ac.UNAVAILABLE_COLUMNS) + " (not measured)")
EST_TITLES = {
    "openhand": "estimated (open-hand calibration; not a validated Paper 3 coordinate)",
    "neutral": ("estimated (tmc_neutral calibration take; not a validated Paper 3 "
                "coordinate)")}
EST_TITLE = EST_TITLES["openhand"]        # the block's title with no calibration loaded
EST_NONE = "not computed (no calibration loaded)"
EST_LABEL = "unvalidated estimate"
# the reference-mapped block's tag, by the mapping's status: what its reference was
MAP_TAGS = {"evaluated": "reference-evaluated",
            "synthetic_demonstration": "SYNTHETIC, not measured",
            "cross_check": "cross-check only"}
C_MAP = "#8fd0ff"
PANEL_CHARS = 83                          # the angle panel's width (see the RECT_ note)
MCP_ORDER_LABELS = {"": "(xyz, flexion first)",
                    "_paper_order": "(paper order, abduction first)"}
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
    "a new run, so a jump is not counted as skipped frames. rendered_updates "
    "= draws that reached the screen: a blit of the changed regions after a "
    "tick, or a full draw (not a saved screenshot). frames_shown = ticks that "
    "displayed a source frame other than the previous tick's (distinct frames "
    "that appeared on screen). wall_duration_s = wall time from the start of "
    "a play run to its last tick, summed over the runs; rendered_per_s and "
    "frames_shown_per_s divide the counts made during the runs by it. "
    "runs_detail gives the same per run.")

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
C_ANG_PANEL = "#0b120e"
C_ANG_TITLE = "#7fe0a0"
C_ANG_EDGE = "#3d8a5c"
STATUS_COLORS = {"applied": "#f0f0f0", "informational": C_CYAN,
                 "not_applied": "#7a7a85", "unavailable": "#5c5c6c"}
ANGLE_COLORS = {"calculated": "#f0f0f0", "outside_paper3": "#9fb6cc",
                "unavailable": "#6c6c7c", "estimated": C_WARN}
EVENT_COLORS = {"gap": "#ff3b3b", "id_change": "#ff9f1a",
                "dropped_frames": "#ffe14d", "invalid_status": "#9a9a9a",
                "clock_defect": "#b36bff", "hold": "#2ecc71"}
TRIAD_COLORS = ("#ff4040", "#40e040", "#4080ff")
OVERLAY_COLOR = "#ffe14d"
RESIDUAL_COLOR = "#ff2020"

HUD_FONT, P3_FONT, ANG_FONT, LINE_SPACING, ANG_SPACING = 6.8, 6.8, 6.4, 1.22, 1.1
NOTE_FONT = 7.0
# figure fractions: [left, bottom, width, height]. At 100 dpi (this machine's
# TkAgg window and the headless screenshots) every panel font is 9 px with a
# 5 px advance: the angle panel holds 83 characters (its widest row is 81) and
# about 57 rows at the window's 804 px height (56 before the panels' bottom moved
# from 0.24 to 0.225 on 2026-10-24, clear of the timeline legend below 0.22).
RECT_3D = [0.005, 0.33, 0.225, 0.55]
RECT_HUD = [0.235, 0.225, 0.22, 0.735]
RECT_ANG = [0.46, 0.225, 0.28, 0.735]
RECT_P3 = [0.745, 0.225, 0.25, 0.735]
RECT_TIMELINE = [0.06, 0.155, 0.90, 0.045]
RECT_SLIDER = [0.06, 0.112, 0.90, 0.025]


def _fmt(v, spec: str, none: str = "n/a") -> str:
    return none if v is None else format(v, spec)


def _deg(v) -> str:
    """Signed degrees to 0.1 for the angle panel, 'n/a' for None, '0.0'
    for a value that rounds to zero."""
    if v is None:
        return "n/a"
    text = f"{v:+.1f}"
    return "0.0" if float(text) == 0.0 else text


def _angle_groups() -> List[Tuple[str, List[Tuple[str, str]]]]:
    """(group, [(label, column)]) for the angle panel, from
    angle_coverage.COVERAGE: the wrist, then each digit, every column of a
    calculated or outside-Paper-3 entry (twist apart: it goes on the digit's
    line). A finger MCP entry has two columns, each labelled with its order."""
    groups: Dict[str, List[Tuple[str, str]]] = {"WRIST": []}
    short = {"flexion/extension": "fe", "abduction/adduction": "aa",
             "radial/ulnar deviation": "deviation"}
    for e in ac.COVERAGE:
        if (e.status not in ("calculated", "outside_paper3")
                or e.movement not in short):
            continue
        digit, joint = (e.joint.split() + [""])[:2]
        what = short[e.movement]
        for column in e.output_columns:
            if digit == "wrist":
                label = f"wrist {what} (forearm to palm)"
            elif len(e.output_columns) == 2:            # finger MCP: both orders
                order = "_paper_order" if column.endswith("_paper_order") else ""
                label = f"{joint} {what} {MCP_ORDER_LABELS[order]}"
            else:
                label = f"{joint} {what}"
            groups.setdefault(digit.upper(), []).append((label, column))
    return list(groups.items())


ANGLE_GROUPS = _angle_groups()


class _QuietSlider(Slider):
    """A Slider that never draws by itself (its constructor would do a full
    draw): the viewer draws once per change, in _draw."""
    drawon = False


class _GlyphAtlas:
    """Every character of one font (size and weight) rendered once by Pillow
    into a cell one advance wide plus the one column a wide glyph spills into
    the next cell, as uint8 coverage; characters are added as they appear.
    `strip` composes a line by indexing the atlas, no per-glyph drawing."""

    def __init__(self, font: ImageFont.FreeTypeFont):
        self.font = font
        ascent, descent = font.getmetrics()
        self.h = ascent + descent
        self.adv = max(1, int(round(font.getlength("M"))))
        self.ids: Dict[str, int] = {}
        self._table = np.full(0x10000, -1, dtype=np.int32)   # UTF-16 code unit -> id
        self._cells: List[np.ndarray] = []
        self._stack: Optional[np.ndarray] = None

    def _add(self, ch: str) -> None:
        img = Image.new("L", (self.adv + 1, self.h), 0)
        ImageDraw.Draw(img).text((0, 0), ch, font=self.font, fill=255)
        self.ids[ch] = len(self._cells)
        if ord(ch) < 0x10000:
            self._table[ord(ch)] = self.ids[ch]
        self._cells.append(np.asarray(img, dtype=np.uint8))
        self._stack = None

    def ids_of(self, text: str) -> np.ndarray:
        """Glyph id per character, rendering the ones not seen yet."""
        units = np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16)
        ids = self._table[units]
        if len(units) != len(text) or (ids < 0).any():   # new glyphs, or beyond the BMP
            for ch in text:
                if ch not in self.ids:
                    self._add(ch)
            if len(units) != len(text):
                return np.array([self.ids[ch] for ch in text], dtype=np.int32)
            ids = self._table[units]
        return ids

    def strip(self, text: str) -> np.ndarray:
        """Coverage (h, len(text) * adv) of one line of text."""
        n = len(text)
        if n == 0:
            return np.zeros((self.h, 0), dtype=np.uint8)
        ids = self.ids_of(text)
        if self._stack is None:
            self._stack = np.stack(self._cells)
        g = self._stack[ids]                                        # (n, h, adv + 1)
        out = np.ascontiguousarray(g[:, :, :self.adv].transpose(1, 0, 2))
        out = out.reshape(self.h, n * self.adv)
        if n > 1:                     # the spill column of glyph i lands on cell i + 1
            cols = out[:, self.adv::self.adv]
            np.maximum(cols, g[:-1, :, self.adv].T, out=cols)
        return out


class _TextPanel(Artist):
    """Coloured monospace lines filling one axes, composed from a glyph atlas
    (`_GlyphAtlas`, Pillow-rendered, matplotlib's monospace font DejaVu Sans
    Mono) and drawn as one bitmap.

    Why not matplotlib Text: matplotlib 3.11's Agg backend lays out and
    renders text glyph by glyph in Python, about 40 us a glyph; the three
    panels hold about 5000 glyphs, 450 ms a frame that way. Why not Pillow's
    own text drawing line by line (the first version): 0.18 ms a line, about
    10 ms a panel and 30 ms a tick for the three, the largest cost of a
    playback tick measured on 2026-10-24. Here each glyph is rendered once;
    a row of text is its coverage from the atlas blended once over the
    panel colour and kept (`_row`), so a tick pays only for the rows whose
    text or colour changed, and the panel is assembled by copying rows into
    one array: under a millisecond for an unchanged panel, a few for one
    whose values all changed. Rendered at draw time at the axes' pixel size
    and the renderer's dpi (cached until the lines or the size change), so a
    resized window or a hi-dpi screen gets crisp text. The first line is
    drawn bold (the title). A row's descenders reach into the next row's
    band (the glyph cell is taller than the line step) and are blended
    there, as Pillow's line-by-line drawing did.
    """

    def __init__(self, ax, fontsize: float, bg: str, pad_px: int = 6,
                 spacing: float = LINE_SPACING):
        super().__init__()
        self._ax, self.fontsize, self.bg, self.pad = ax, fontsize, bg, pad_px
        self.spacing = spacing
        self.lines: List[Tuple[str, str]] = []
        self._fonts: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}
        self._atlases: Dict[Tuple[int, bool], _GlyphAtlas] = {}
        self._wrapped: Dict[Tuple[str, int], List[str]] = {}
        self._rows: Dict[tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._rgb: Dict[str, np.ndarray] = {}
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

    def _atlas(self, px: int, bold: bool) -> _GlyphAtlas:
        if (px, bold) not in self._atlases:
            self._atlases[(px, bold)] = _GlyphAtlas(self._font(px, bold))
        return self._atlases[(px, bold)]

    def _wrap(self, text: str, width: int) -> List[str]:
        key = (text, width)
        if key not in self._wrapped:
            if len(self._wrapped) > 4096:
                self._wrapped.clear()
            self._wrapped[key] = textwrap.wrap(text, width, subsequent_indent="    ") or [""]
        return self._wrapped[key]

    def _color(self, name: str) -> np.ndarray:
        if name not in self._rgb:
            self._rgb[name] = np.array(ImageColor.getrgb(name)[:3], dtype=np.float32)
        return self._rgb[name]

    def _lut(self, color: str) -> np.ndarray:
        """(256,) uint32: the RGBA pixel (packed as it lies in memory) of
        `color` over the panel colour at each coverage value: the blend as one
        table lookup per pixel."""
        key = ("lut", color)
        lut = self._rows.get(key)
        if lut is None:
            k = np.arange(256, dtype=np.float32)[:, None] * (1.0 / 255.0)
            bg, rgb = self._color(self.bg), self._color(color)
            lut8 = np.empty((256, 4), dtype=np.uint8)
            lut8[:, :3] = (bg + (rgb - bg) * k + 0.5).astype(np.uint8)
            lut8[:, 3] = 255
            lut = self._rows[key] = lut8.view(np.uint32).reshape(256)
        return lut

    def _row(self, atlas: _GlyphAtlas, piece: str, color: str, px: int, bold: bool,
             step: int):
        """(the row's pixels (cell_h, W) as packed uint32 blended over the panel
        colour, the coverage of its descenders below the line step or None,
        colour rgb as uint16) of one row of text, kept for the next render: a
        tick pays only for the rows that changed."""
        key = (piece, color, px, bold)
        hit = self._rows.get(key)
        if hit is None:
            if len(self._rows) > 4096:
                self._rows.clear()
            cov = atlas.strip(piece)
            below = cov[step:]
            hit = self._rows[key] = (
                self._lut(color)[cov],
                below.astype(np.uint16) if below.size and below.any() else None,
                self._color(color).astype(np.uint16))
        return hit

    def _render(self, w: int, h: int, dpi: float) -> np.ndarray:
        px = max(6, int(round(self.fontsize * dpi / 72.0)))
        step = int(round(px * self.spacing))
        atlas, bold = self._atlas(px, False), self._atlas(px, True)
        width = max(20, int((w - 2 * self.pad) // atlas.adv))
        img32 = np.empty((h, w), dtype=np.uint32)
        img32[:] = self._lut(self.bg)[0]
        img = img32.view(np.uint8).reshape(h, w, 4)
        pad, y, self.rows, self.truncated = self.pad, self.pad, 0, False
        hang = None                     # the previous row's descenders, for the next band
        for k, (text, color) in enumerate(self.lines):
            for piece in self._wrap(text, width):
                if y + step > h - pad:
                    self.truncated = True
                    break
                row, below, rgb = self._row(bold if k == 0 else atlas, piece, color, px,
                                            k == 0, step)
                ww = min(row.shape[1], w - pad)
                rows_c = min(step, row.shape[0])
                img32[y:y + rows_c, pad:pad + ww] = row[:rows_c, :ww]
                if hang is not None:    # blend the row above's descenders over this band
                    self._hang(img, hang, y, pad, w)
                hang = None if below is None else (below, rgb)
                y += step
                self.rows += 1
        if hang is not None and y < h:
            self._hang(img, hang, y, pad, w)
        return img

    @staticmethod
    def _hang(img, hang, y, pad, w) -> None:
        """Blend a row's descenders (coverage below its line step, uint16)
        over the band of the row below, in integer arithmetic."""
        cov, rgb = hang
        d = min(cov.shape[0], img.shape[0] - y)
        ww = min(cov.shape[1], w - pad)
        if d <= 0 or ww <= 0:
            return
        c = cov[:d, :ww, None]
        region = img[y:y + d, pad:pad + ww, :3]
        region[...] = ((region.astype(np.uint16) * (255 - c) + rgb * c + 127) // 255
                       ).astype(np.uint8)

    def draw(self, renderer) -> None:
        if not self.get_visible():
            return
        box = self._ax.get_window_extent(renderer)
        # one pixel inside the axes: the frame's spines stay in the background
        x0, y0 = int(round(box.x0)) + 1, int(round(box.y0)) + 1
        w, h = int(round(box.x1)) - x0 - 1, int(round(box.y1)) - y0 - 1
        if w < 2 or h < 2:
            return
        key = (tuple(self.lines), w, h, renderer.dpi)
        if key != self._key:
            self._rgba, self._key = self._render(w, h, renderer.dpi), key
        buffer = getattr(renderer, "buffer_rgba", None)
        if buffer is not None:
            # Agg: the panel is opaque, so copy it straight into the renderer's
            # buffer (row 0 at the top) instead of alpha-compositing it through
            # draw_image (0.8 ms a panel against 0.15; measured 2026-10-24)
            buf = np.asarray(buffer())
            top = buf.shape[0] - (y0 + h)
            if top >= 0 and y0 >= 0 and x0 >= 0 and x0 + w <= buf.shape[1]:
                buf[top:top + h, x0:x0 + w] = self._rgba
                self.stale = False
                return
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

    tmc_provider: optional callable (pos, rot, hand_side) -> dict with `kind`
    (one of `tmc_estimate.KINDS`; "openhand" when absent), that kind's two
    estimate columns (`angle_coverage.ESTIMATE_COLUMNS_BY_KIND[kind]`, degrees
    or None), a `provenance` string, and optionally its twist and residual
    (`tmc_estimate.keys_for(kind)[2:]`, shown on one line when the twist is
    given), called on the displayed pose; its numbers go to the angle panel's
    "estimated" block, titled by the kind, and nowhere else. Without one that
    block says "not computed".

    mapping_provider: optional callable (pos, rot, hand_side) -> dict with
    T_TMC_fe_mapped / T_TMC_aa_mapped (degrees or None), `status` (a row
    status of leap_hand.reference_validation.ROW_STATUSES, or "not_applied"),
    `mapping_status` (the mapping's: evaluated, synthetic_demonstration or
    cross_check), `note` (why there is no value) and `scope` (the mapping's
    scope text), called on the displayed pose; shown in the
    "reference-mapped" block only.
    """

    def __init__(self, rec: rc.Recording, *, mode: str = "hold",
                 reference: str = "camera", speed: float = 1.0,
                 start_frame: int = 0, overlay: bool = False,
                 axes: bool = False, figsize=(15.5, 8.6), dpi=None,
                 now=time.perf_counter,
                 screenshot_dir="results/replay_fidelity/screens",
                 timing_label: Optional[str] = None, tmc_provider=None,
                 mapping_provider=None):
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
        self.tmc_provider = tmc_provider
        self.mapping_provider = mapping_provider
        self._ang_cache: Dict[int, Tuple[dict, Optional[dict], Optional[dict]]] = {}
        # scheduling: one record per tick; skipped frames counted per play run
        self.sched = rc.SchedulingLog()
        self._tick_run: List[int] = []       # play run of each tick
        self._runs = 0
        self._last_shown: Optional[int] = None
        self._frames_skipped = 0
        self._next_intended: Optional[float] = None
        self._summary: Optional[dict] = None
        self._rendered = 0                   # draws that reached the screen
        self._run_stats: List[dict] = []     # one record per play run (TIMING_DOC)
        self._run: Optional[dict] = None     # the run in progress
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
        self._ang_lines: List[Tuple[str, str]] = []
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
                                      color=C_DIM, family="monospace",
                                      fontsize=NOTE_FONT, va="top", ha="left")
        w_in = self.fig.get_size_inches()[0]
        self._ref_width = int(RECT_3D[2] * w_in * 72 / (7.5 * 0.602)) - 2
        self._note_width = int(RECT_3D[2] * w_in * 72 / (NOTE_FONT * 0.602)) - 2

        # panels: HUD (fidelity), angles, Paper 3 (research), visibly apart
        self.hud_ax = self._panel(RECT_HUD, "#2a2a35")
        self.ang_ax = self._panel(RECT_ANG, C_ANG_EDGE)
        self.p3_ax = self._panel(RECT_P3, C_P3_EDGE)
        self.hud_panel = _TextPanel(self.hud_ax, HUD_FONT, C_PANEL)
        self.ang_panel = _TextPanel(self.ang_ax, ANG_FONT, C_ANG_PANEL,
                                    spacing=ANG_SPACING)
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
        # restores the background and redraws only those (_draw_dynamic). The
        # 3D axes itself is not animated: its panes, grid, ticks and labels
        # never move and stay in the background; only the hand's artists do.
        # Likewise the panels' frames and the slider's track and label stay;
        # the panel bitmaps, the slider's bar, handle and value, and the
        # timeline cursor move.
        self._hand_artists = [self._palm, self._bones, *self._triads, self._residuals,
                              self._joints, self._rec_markers, self._tips]
        slider_parts = [self.slider.poly, self.slider.valtext]
        if getattr(self.slider, "_handle", None) is not None:
            slider_parts.append(self.slider._handle)
        self._dynamic_2d = [self.hud_panel, self.ang_panel, self.p3_panel, *slider_parts,
                            self._cursor]
        self._dynamic = self._hand_artists + self._dynamic_2d
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
        self._ang_lines = self._compose_angles()
        self.ang_panel.set_lines(self._ang_lines)
        self._cursor.set_xdata([self.frame_index, self.frame_index])
        self._sync_widgets()
        canvas = self.fig.canvas
        if self._bg is not None and self._bg_key == self._static_key():
            canvas.restore_region(self._bg)
            self._draw_dynamic(canvas.get_renderer())
            for box in self._blit_boxes():
                canvas.blit(box)
            self._count_render()
        else:                               # something static changed: full draw
            canvas.draw_idle()

    def _draw_dynamic(self, renderer) -> None:
        """The animated artists over the background: the hand's 3D artists,
        projected with the axes' current matrix and drawn far to near (as
        Axes3D.draw orders every collection), then the text panels, the
        slider and the timeline cursor."""
        ax = self.ax
        ax.M = ax.get_proj()
        ax.invM = np.linalg.inv(ax.M)
        depth = [(a.do_3d_projection(), a) for a in self._hand_artists if a.get_visible()]
        for _z, a in sorted(depth, key=lambda d: d[0] if d[0] == d[0] else -np.inf,
                            reverse=True):
            a.draw(renderer)
        for a in self._dynamic_2d:
            a.draw(renderer)

    def _blit_boxes(self) -> List[Bbox]:
        """The screen region a tick can change, as one box: the union of the
        3D axes (padded, the markers are wider than a point), the three
        panels, the slider and the timeline; about 85 % of the figure. One
        blit of it measured 10.6 ms on this machine against 14.5 ms for the
        whole figure and 12.3 ms for the four regions blitted one by one
        (the Tk photo update costs per call as well as per pixel)."""
        return [Bbox.union([self.ax.bbox.expanded(1.15, 1.15), self.hud_ax.bbox,
                            self.ang_ax.bbox, self.p3_ax.bbox, self.slider_ax.bbox,
                            self.tl_ax.bbox])]

    def _count_render(self) -> None:
        self._rendered += 1
        if self._run is not None:
            self._run["rendered_updates"] += 1

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
        self._draw_dynamic(event.renderer)
        if not canvas.is_saving():
            self._count_render()

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
        self._note_3d.set_text("\n".join(piece for note in notes
                                         for piece in textwrap.wrap(note, self._note_width)))

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
            out.append((f"rendering: {s['rendered_updates']} updates; over "
                        f"{s['wall_duration_s']:.2f} s of play "
                        f"{_fmt(s['rendered_per_s'], '.1f')} updates/s, "
                        f"{_fmt(s['frames_shown_per_s'], '.1f')} distinct frames/s",
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

    def _angles_of_displayed(self) -> Tuple[dict, Optional[dict], Optional[dict]]:
        """(angle_coverage.all_angles, the tmc_provider's dict or None, the
        mapping_provider's dict or None) of the displayed pose; cached per
        source frame, an interpolated pose computed each time."""
        def compute(pos, rot):
            side = self.rec.hand_side
            est = None if self.tmc_provider is None else self.tmc_provider(pos, rot, side)
            mapped = (None if self.mapping_provider is None
                      else self.mapping_provider(pos, rot, side))
            return ac.all_angles(pos, rot, side), est, mapped
        d = self.displayed
        if d.source in ("recorded", "held"):
            if d.i0 not in self._ang_cache:
                r = self._recons[d.i0]
                self._ang_cache[d.i0] = compute(r.pos, r.rot)
            return self._ang_cache[d.i0]
        return compute(d.pos, d.rot)

    def _compose_angles(self) -> List[Tuple[str, str]]:
        """The angle panel: wrist first, then one block per digit (its line
        carries the twist of each of its bones), every column the coverage
        table gives a value with that value and its status; the six
        unavailable named; the estimate block apart (module docstring)."""
        d = self.displayed
        out = [(ANG_TITLE, C_ANG_TITLE), (ANG_SIGNS, C_TEXT), (ANG_TWIST, C_DIM)]
        if d.pos is None:
            out.append(("no pose at this media time", C_WARN))
            return out
        if d.source == "interpolated":
            a, b = self.rec.frames[d.i0], self.rec.frames[d.i1]
            out.append((f"interpolated between frame_id {a.frame_id} and "
                        f"{b.frame_id} (not a recorded pose)", C_WARN))
            where = "interpolated pose"
        else:
            where = f"source frame_id {d.frame.frame_id}"
        angles, est, mapped = self._angles_of_displayed()

        def row(label, column):
            status = ac.status_of(column)
            return (f"{label:<37} {column:<20} {_deg(angles[column]):>6} "
                    f"{ac.STATUS_LABELS[status]}", ANGLE_COLORS[status])

        out.append((f"{where:<37} {'column':<20} {'value':>6} status", C_DIM))
        for group, rows in ANGLE_GROUPS:
            if group != "WRIST":
                twist = [(j.split("_", 1)[1][:4], angles[f"{j}_twist_tracker"])
                         for j in ac.TWIST_JOINTS if j.startswith(group + "_")]
                out.append((f"{group:<7}twist " + " ".join(
                    f"{bone:<4} {_deg(v):>5}" for bone, v in twist), C_CYAN))
            out.extend(row(label, column) for label, column in rows)
        out.append((ANG_UNAVAILABLE, ANGLE_COLORS["unavailable"]))
        out.extend(self._mapped_rows(mapped))
        out.append(("", C_TEXT))
        kind = (est or {}).get("kind") or "openhand"
        if kind not in te.KINDS:
            raise ValueError(f"tmc_provider gave kind {kind!r}, expected one of {te.KINDS}")
        twist_key, residual_key = te.keys_for(kind)[2:]
        out.append((EST_TITLES[kind], C_WARN))
        for column in ac.ESTIMATE_COLUMNS_BY_KIND[kind]:
            if est is None:
                text = EST_NONE
            elif est.get(column) is None:
                text = "not computed (the provider gave no value)"
            else:
                text = f"{_deg(est[column])} deg ({EST_LABEL})"
            out.append((f"{column}: {text}", C_WARN))
        if est is not None:
            if est.get(twist_key) is not None:
                out.append((f"{twist_key}: {_deg(est[twist_key])} deg (residual "
                            f"{_fmt(est.get(residual_key), '.1f')} deg)", C_WARN))
            out.append((f"provenance: {est.get('provenance') or '(none given)'}", C_DIM))
        return out

    @staticmethod
    def _mapped_rows(mapped: Optional[dict]) -> List[Tuple[str, str]]:
        """The reference-mapped block (class docstring, mapping_provider): one
        row with the tag and the two values (or why they are empty), one with
        the scope (or why not applied), each cut to the panel's width so the
        block is two rows whatever it says. Nothing without a provider."""
        if mapped is None:
            return []
        fe, aa = (mapped.get(c) for c in ac.MAPPED_COLUMNS)
        head = f"mapped ({MAP_TAGS[mapped['mapping_status']]}): "
        if mapped.get("status") == "not_applied":
            first, second = head + "not applied to this recording", mapped.get("note")
        elif fe is None or aa is None:
            first, second = head + "empty here", mapped.get("note")
        else:
            first = (f"{head}{ac.MAPPED_COLUMNS[0]} {_deg(fe)} {ac.MAPPED_COLUMNS[1]} "
                     f"{_deg(aa)}")
            second = mapped.get("scope_row") or f"scope {mapped.get('scope')}"
        cut = lambda s: textwrap.shorten(str(s), PANEL_CHARS, placeholder=" ...")  # noqa
        return [(cut(first), C_MAP), (cut(second), C_DIM)]

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
        wall = float(self._now() if now is None else now)
        self._run = {"run": self._runs, "speed": self.clock.speed,
                     "media_start_s": self.clock.media_time(now),
                     "media_end_s": self.clock.media_time(now),
                     "wall_start_s": wall, "wall_last_s": wall, "ticks": 0,
                     "rendered_updates": 0, "frames_shown": 0, "frames_skipped": 0}
        self._run_stats.append(self._run)

    def _pause_clock(self, now) -> float:
        """Pause, clamped to the end of the recording; the media time."""
        self.clock.pause(now)
        self._run = None
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
        run = self._run                    # the run this tick belongs to, even its last
        if media >= float(self.t[-1]):
            media = self._pause_clock(t_now)
            self._status = "end of the recording: paused"
            self._run = run                # the final draw still counts for this run
        self.displayed = rc.pose_at(self.rec, media, self.mode)
        self.media_t = media
        if self.displayed.i0 is not None:
            self.frame_index = self.displayed.i0
        self.sched.record(self._next_intended, reached, self.frame_index)
        self._tick_run.append(self._runs)
        skipped = 0
        if self._last_shown is not None:
            skipped = max(0, self.frame_index - self._last_shown - 1)
            self._frames_skipped += skipped
        if run is not None:
            run["ticks"] += 1
            run["wall_last_s"] = t_now
            run["media_end_s"] = media
            run["frames_skipped"] += skipped
            if self.frame_index != self._last_shown:
                run["frames_shown"] += 1
        self._last_shown = self.frame_index
        self._next_intended = reached + self.interval_s * self.clock.speed
        self._summary = None
        self._draw()
        if not self.playing:
            self._run = None
        return self.displayed

    def _runs_detail(self) -> List[dict]:
        out = []
        for r in self._run_stats:
            wall = r["wall_last_s"] - r["wall_start_s"]
            out.append({**{k: v for k, v in r.items()
                           if k not in ("wall_start_s", "wall_last_s")},
                        "wall_duration_s": wall,
                        "rendered_per_s": r["rendered_updates"] / wall if wall > 0 else None,
                        "frames_shown_per_s": r["frames_shown"] / wall if wall > 0 else None})
        return out

    def timing_summary(self) -> dict:
        """SchedulingLog.stats() over every tick (n, mean_ms, p95_ms, max_ms),
        frames_skipped counted inside play runs (the log's own count would
        pair the last tick before a seek with the first one after it), the
        rendering counts (rendered_updates, frames_shown, wall_duration_s,
        rendered_per_s, frames_shown_per_s; TIMING_DOC) in all and per run
        (runs_detail), plus what is logged, the speed history and the
        backend."""
        if self._summary is None:
            self._summary = {**self.sched.stats(),
                             "frames_skipped": self._frames_skipped}
        runs = self._runs_detail()
        wall = sum(r["wall_duration_s"] for r in runs)
        rendered = sum(r["rendered_updates"] for r in runs)
        shown = sum(r["frames_shown"] for r in runs)
        return {**self._summary, "runs": self._runs,
                "rendered_updates": self._rendered, "frames_shown": shown,
                "wall_duration_s": wall,
                "rendered_per_s": rendered / wall if wall > 0 else None,
                "frames_shown_per_s": shown / wall if wall > 0 else None,
                "runs_detail": runs,
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

    def angles_text(self) -> str:
        """The angle panel as shown (unwrapped lines)."""
        return "\n".join(text for text, _ in self._ang_lines)

    def panel_texts(self) -> Dict[str, str]:
        """The three text panels as shown: hud, angles, paper3."""
        return {"hud": self.hud_text(), "angles": self.angles_text(),
                "paper3": self.paper3_text()}

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
