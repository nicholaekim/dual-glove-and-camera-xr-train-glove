"""Per pose: the photo, and the Leap and glove hand models each nearest the
photo's moment, and the same 21 points from both in one frame.

Reads a session recorded by `scripts/record_simultaneous.py --camera leap`
(`leap/<take>.jsonl` + `.meta.json`, `glove/<take>.jsonl`, `stills/<take>.jpg`)
and for every accepted take writes, under `--out` (default
`results/paired_pose/<session name>/`):

  <take>/camera.txt               the camera's frame nearest the moment, professor format
  <take>/glove.txt                the glove's frame nearest the moment, professor format
  <take>/points_common_frame.csv  21 points of both sensors in one frame + distance
  <take>/bone_lengths.csv         20 bones, glove and camera
  <take>/notes.txt                the moment, both time offsets, labels, handedness
  <take>/figure.png               photo | Leap model | glove model
  comparison.csv                  one row per take, refused ones with the reason
  figures.pdf                     every figure.png, one page per take
  README.md                       what each file is and every convention

The moment is the photo's (`leap_hand.paired_pose.take_moment`); each
sensor's frame is the one nearest it on the capture clock, within
`--max-dt-ms`. No lag correction, no scaling, no fit. Coordinates only.

Usage:
  python scripts/paired_pose_compare.py recordings/sync_day2
  python scripts/paired_pose_compare.py <session> --takes fist_left_take1_<stamp> ...
  python scripts/paired_pose_compare.py <session> --include-rejected --out somewhere
  python scripts/paired_pose_compare.py --check-outputs results/paired_pose/<session>
"""
import argparse
import csv
import datetime
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402
import numpy as np                                                 # noqa: E402

from cam_hand.prof_format import parse_text                        # noqa: E402
from leap_hand.paired_pose import (BONES, CAPTURE_CLOCK, CHAINS, HANDEDNESS_MARGIN_MM,  # noqa: E402
                                   INDETERMINATE, LABEL, OPPOSITE, SAME, SIMULTANEITY,
                                   TIMING_LABEL,
                                   SPEED_STEP_S, STILL_CAPTURE, STILL_MM_S, STILL_MTIME,
                                   STILL_WINDOW_S, DegeneratePalm, axis_definition,
                                   bone_lengths, compare_points, find_meta, frame_nearest,
                                   handedness, load_frames, palm_side_mm, points_mm,
                                   still_file, stillness, take_moment, to_palm_frame,
                                   window_speed_mm_s)

REPO = Path(__file__).resolve().parents[1]
GLOVE_COLOUR, CAMERA_COLOUR = "#eb6834", "#2a78d6"     # the project's chart colours
OK = "ok"
TAKE_FILES = ("camera.txt", "glove.txt", "points_common_frame.csv", "bone_lengths.csv",
              "notes.txt", "figure.png")
COLUMNS = ["take", "pose", "hand", "moment_source", "cam_dt_ms", "glove_dt_ms",
           "handedness", "glove_z_margin_mm", "camera_z_margin_mm",
           "glove_speed_mm_s", "camera_speed_mm_s", "stillness",
           "median_mm", "max_mm", "max_point", "n_points", "status"]
STAT_BASIS = "median and maximum over the 20 non-wrist landmarks"
POINT_COLUMNS = ["index", "name", "glove_x_mm", "glove_y_mm", "glove_z_mm",
                 "camera_x_mm", "camera_y_mm", "camera_z_mm", "distance_mm"]
LETTERS = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "pinky": "L"}
# Measured 5 October 2026 on closing movements (glove against camera); not corrected here.
GLOVE_LAG = {"left": "about 0.47 s", "right": "about 0 s"}


def glove_exporter():
    """`scripts/glove/export_prof_format.py`, imported by path: the one
    definition of the professor's block (as scripts/leap/record_frame.py)."""
    path = REPO / "scripts" / "glove" / "export_prof_format.py"
    spec = importlib.util.spec_from_file_location("glove_prof_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


# --- one take ---------------------------------------------------------------------
def process_take(session: Path, stem: str, meta: dict, meta_path: Path, out: Path,
                 max_dt_s: float, exporter):
    """(comparison row, figure or None). The row's status is "ok" or why not."""
    hand = meta.get("hand")
    row = {"take": stem, "pose": meta.get("pose"), "hand": hand, "status": ""}
    cam_path = meta_path.parent / f"{stem}.jsonl"
    glove_path = meta_path.parent.parent / "glove" / f"{stem}.jsonl"
    if not cam_path.is_file():
        return dict(row, status=f"no camera file ({cam_path.name})"), None
    if not glove_path.is_file():
        return dict(row, status=f"no glove file ({glove_path.name})"), None
    moment = take_moment(session, stem, meta)
    if moment is None:
        still = still_file(session, meta)
        return dict(row, status="no moment: no still_capture_time in the meta and no still "
                                f"file ({still if still else 'no still in the meta'})"), None
    row["moment_source"] = moment.source

    cam_all = load_frames(cam_path)
    glove_all = load_frames(glove_path)
    glove = [f for f in glove_all if f.line.get("hand_side") == hand]
    picked = {}
    for name, frames, what in (("cam", cam_all, "camera frame"),
                               ("glove", glove, f"glove frame of the {hand} hand")):
        near = frame_nearest(frames, moment.t, math.inf)
        if near is None:
            return dict(row, status=f"no {what} in the file"), None
        if abs(near.dt) > max_dt_s:
            return dict(row, status=f"no {what} within {max_dt_s * 1000:.0f} ms of the "
                                    f"moment (nearest {near.dt * 1000:+.1f} ms)"), None
        picked[name] = near
        row[f"{name}_dt_ms"] = round(near.dt * 1000.0, 1)
    cam, glv = picked["cam"], picked["glove"]

    try:
        g_local, g_basis = to_palm_frame(points_mm(glv.frame), "glove")
        c_local, c_basis = to_palm_frame(points_mm(cam.frame), "camera")
    except DegeneratePalm as why:
        return dict(row, status=str(why)), None
    rows, summary = compare_points(g_local, c_local)
    hand_diag = handedness(g_local, c_local)
    g_side, c_side = palm_side_mm(g_local), palm_side_mm(c_local)
    g_speed, c_speed = sensor_speed(glove, glv), sensor_speed(cam_all, cam)
    still = stillness(g_speed, c_speed)
    row.update(handedness=hand_diag, glove_z_margin_mm=round(g_side, 1),
               camera_z_margin_mm=round(c_side, 1), glove_speed_mm_s=_r1(g_speed),
               camera_speed_mm_s=_r1(c_speed), stillness=still,
               median_mm=round(summary["median_mm"], 1),
               max_mm=round(summary["max_mm"], 1), max_point=summary["max_point"],
               n_points=summary["n_points"], status=OK)
    # The size the still was saved at; never a CSV column, read by the README.
    row["_still_px"] = None

    take_dir = out / stem
    take_dir.mkdir(parents=True, exist_ok=True)
    # ONE block each, the frame nearest the moment in its native frame, stamped
    # with that frame's own time on the clock it was picked on.
    (take_dir / "camera.txt").write_text(exporter.frame_block(cam.frame, cam.time) + "\n",
                                         encoding="utf-8")
    (take_dir / "glove.txt").write_text(exporter.frame_block(glv.frame, glv.time) + "\n",
                                        encoding="utf-8")
    with open(take_dir / "points_common_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(POINT_COLUMNS)
        for r in rows:
            w.writerow([r["index"], r["name"]] + [f"{r[k]:.3f}" for k in
                       ("gx", "gy", "gz", "cx", "cy", "cz", "distance_mm")])
    with open(take_dir / "bone_lengths.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bone", "glove_mm", "camera_mm"])
        for (name, gl), (_, cl) in zip(bone_lengths(g_local), bone_lengths(c_local)):
            w.writerow([name, f"{gl:.2f}", f"{cl:.2f}"])

    still_px = None
    if moment.still is not None:
        img = plt.imread(moment.still)
        still_px = row["_still_px"] = (img.shape[1], img.shape[0])
    else:
        img = None
    cam_line, glv_line = cam_all[cam.index].line, glove[glv.index].line
    labels = {"camera": Counter(f.line.get("hand_side") for f in cam_all),
              "glove": Counter(f.line.get("hand_side") for f in glove_all)}
    fr = [f.line.get("framerate") for f in cam_all]
    fr = [float(v) for v in fr if isinstance(v, (int, float)) and v > 0]
    if fr:
        rate = (float(np.median(fr)), "median `framerate` of this take's camera frames")
    else:
        ct = np.diff(np.sort([f.capture_time if cam.clock == CAPTURE_CLOCK else f.wall_time
                              for f in cam_all]))
        ct = ct[ct > 0]
        rate = ((1.0 / float(np.median(ct)), f"median spacing of this take's camera frames "
                 f"on {cam.clock}") if ct.size else (float("nan"), "no frame spacing"))
    info = dict(stem=stem, meta=meta, moment=moment, still_px=still_px, cam=cam, glv=glv,
                cam_line=cam_line, glv_line=glv_line, labels=labels, hand_diag=hand_diag,
                glove_side=g_side, cam_side=c_side, g_basis=g_basis, c_basis=c_basis,
                g_speed=g_speed, c_speed=c_speed, still=still, rate=rate, summary=summary,
                max_dt_s=max_dt_s)
    (take_dir / "notes.txt").write_text(take_notes(info), encoding="utf-8")

    fig = draw_figure(img, g_local, c_local, hand, meta, stem, moment, cam, glv,
                      cam_line.get("hand_side"), glv_line.get("hand_side"), hand_diag, still,
                      summary)
    fig.savefig(take_dir / "figure.png", dpi=100)
    return row, fig


def _r1(v: float):
    return round(v, 1) if np.isfinite(v) else None


def sensor_speed(frames, near) -> float:
    """`window_speed_mm_s` of one sensor's frames around its selected frame,
    on the clock that frame was picked on."""
    t = np.array([f.capture_time if near.clock == CAPTURE_CLOCK else f.wall_time
                  for f in frames], float)
    keep = np.flatnonzero(np.abs(t - near.time) <= STILL_WINDOW_S + SPEED_STEP_S)
    pts = np.array([points_mm(frames[k].frame) for k in keep]).reshape(-1, 21, 3)
    return window_speed_mm_s(t[keep], pts, near.time)


def _counts(c: Counter) -> str:
    return ", ".join(f"{k} {n} lines" for k, n in sorted(c.items(), key=lambda kv: str(kv[0])))


def _basis_rows(basis) -> list:
    return [f"    {axis} = ({', '.join(f'{v:+.3f}' for v in basis[:, k])})"
            for k, axis in enumerate("xyz")]


def take_notes(i: dict) -> str:
    meta, moment, cam, glv = i["meta"], i["moment"], i["cam"], i["glv"]
    cam_line, glv_line, summary = i["cam_line"], i["glv_line"], i["summary"]
    hand = meta.get("hand")
    lag = GLOVE_LAG.get(hand, "not measured")
    lines = [
        f"take: {i['stem']}",
        f"pose: {meta.get('pose')}   take number: {meta.get('take')}   hand: {hand} "
        "(the glove's label names the hand)",
        f"result: {LABEL}; {TIMING_LABEL}. Each sensor's palm frame is built from its own "
        "landmarks, so a difference in the hand's orientation between the sensors is removed, "
        "not measured; the two sensors are not physically registered.",
        SIMULTANEITY,
        "",
        f"moment: {local_iso(moment.t)} local time ({moment.t:.3f} s since 1970)",
        f"  source: {moment.source}",
    ]
    if moment.source == STILL_CAPTURE:
        lines.append("  the image frame's own LeapC timestamp converted to the wall clock the same "
                     "way as the tracking frames' capture_time, so still and tracking frame are "
                     "matched on the camera's clock")
    else:
        lines.append("  this is when the JPEG finished writing, a little after its IR image was "
                     "captured; that write delay is unknown (not recorded in this take) and adds "
                     "to every dt below")
    lines.append(f"  still: {moment.still if moment.still else 'not on disk'}"
                 + (f" ({i['still_px'][0]} x {i['still_px'][1]} px, shown as saved)"
                    if i["still_px"] else ""))
    lines += [
        "",
        "dt = the frame's own time minus the moment (negative: the frame is before the photo); "
        f"frames further than {i['max_dt_s'] * 1000:.0f} ms are refused",
        f"camera frame: dt {cam.dt * 1000:+.1f} ms on {cam.clock}; frame_id "
        f"{cam_line.get('frame_id')}, timestamp_us {cam_line.get('timestamp_us')}, capture_time "
        f"{cam_line.get('capture_time')}, hand_id {cam_line.get('hand_id')}, visible_time_us "
        f"{cam_line.get('visible_time_us')}, label at this frame: {cam_line.get('hand_side')}",
        f"  labels in the camera file: {_counts(i['labels']['camera'])} (the tracker's guess, "
        "not used to name the hand)",
        f"glove frame:  dt {glv.dt * 1000:+.1f} ms on {glv.clock}; packet_counter "
        f"{glv_line.get('packet_counter')}, timestamp {glv_line.get('timestamp')}, capture_time "
        f"{glv_line.get('capture_time')}, label at this frame: {glv_line.get('hand_side')}",
        f"  labels in the glove file: {_counts(i['labels']['glove'])}",
        "clocks: both sensors' capture_time are on the PC wall clock (camera: time.time() at "
        "the LeapC callback minus the frame's age; glove: the packet's arrival at the OSC "
        "receiver), so the two dt values are comparable. The glove's time is ARRIVAL, after "
        f"the glove's transmission delay ({lag} for the {hand} glove, measured 5 October 2026 "
        "on closing movements): the glove frame's dt is to arrival, not to the hand.",
    ]
    if cam.clock != CAPTURE_CLOCK or glv.clock != CAPTURE_CLOCK:
        lines.append("  a frame here was picked on wall_time (the JSONL write time), not "
                     "capture_time: the clock statement above does not hold for it")
    rate, rate_src = i["rate"]
    lines += [
        f"timing uncertainty (separate from dt): camera one frame period, "
        f"{1000.0 / rate:.1f} ms (1000 / {rate:.1f} frames/s, {rate_src}), plus the frame-age "
        "precision (not measured; the two clock reads in the callback are microseconds apart); "
        f"glove the transmission delay above ({lag}), not corrected"
        + ("; the moment: plus the unknown still write delay" if moment.source == STILL_MTIME
           else ""),
        "",
        f"stillness: {i['still']} (mean 21-point speed of the wrist-centred points over the "
        f"frames within {STILL_WINDOW_S:g} s of the selected frame: glove "
        f"{_fmt_speed(i['g_speed'])}, camera {_fmt_speed(i['c_speed'])}; \"moving\" above "
        f"{STILL_MM_S:g} mm/s, the speed leap_hand.transition calls settled; the take is "
        "processed either way)",
        "",
        "palm frame bases, each in its sensor's own wrist-centred frame (rows x, y, z, unit "
        "vectors):",
        "  glove:",
    ] + _basis_rows(i["g_basis"]) + ["  camera:"] + _basis_rows(i["c_basis"]) + [
        "  transformation: local = (p - p_WRIST) . [x y z], each coordinate the dot product of "
        "the sensor's wrist-centred point with that sensor's x, y and z above",
        "",
        f"handedness: {i['hand_diag']} (mean palm-normal coordinate z of points 1-20: glove "
        f"{i['glove_side']:+.1f} mm, camera {i['cam_side']:+.1f} mm; no verdict under "
        f"{HANDEDNESS_MARGIN_MM:g} mm; a diagnostic, nothing is reflected or corrected)",
        f"distance between the two sensors' points in the palm frames, {STAT_BASIS}: median "
        f"{summary['median_mm']:.1f} mm, largest {summary['max_mm']:.1f} mm at "
        f"{summary['max_point']} (WRIST is the origin of both and is not counted)",
        "",
        "not applied: any scaling of either hand (the glove's bones are a template hand, the "
        "tracker's change with the pose; see bone_lengths.csv); the unscaled distances are "
        "the result",
        "",
    ]
    return "\n".join(lines)


def _fmt_speed(v: float) -> str:
    return f"{v:.1f} mm/s" if np.isfinite(v) else "not measured (no pair of frames)"


# --- the figure ---------------------------------------------------------------------
def draw_figure(img, g_local, c_local, hand, meta, stem, moment, cam, glv, cam_label,
                glove_label, hand_diag, still, summary):
    """Photo | Leap model | glove model; both models in the same palm frame
    with identical limits and scale. Palm view: vertical = x (fingers up),
    horizontal = y for a left hand and -y for a right one, so both are seen
    from the palm. Side view: vertical = x, horizontal = -z, seen from the
    thumb side."""
    sign = 1.0 if hand == "left" else -1.0
    hands = {"camera": np.asarray(c_local), "glove": np.asarray(g_local)}
    allp = np.vstack(list(hands.values()))
    xr = (allp[:, 0].min() - 14.0, allp[:, 0].max() + 16.0)
    yr = ((sign * allp[:, 1]).min() - 16.0, (sign * allp[:, 1]).max() + 16.0)
    zr = ((-allp[:, 2]).min() - 14.0, (-allp[:, 2]).max() + 14.0)
    hspan, sspan = yr[1] - yr[0], zr[1] - zr[0]

    fig = plt.figure(figsize=(18, 6), dpi=100)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.05, hspan / 230, sspan / 230,
                                              hspan / 230, sspan / 230],
                          left=0.015, right=0.99, top=0.75, bottom=0.10, wspace=0.18)
    ax = fig.add_subplot(gs[0])
    if img is not None:
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    else:
        ax.text(0.5, 0.5, "no still on disk for this take", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(moment.still.name if moment.still else "", fontsize=8, color="#555555")

    panels = (("Leap hand model (reconstructed)", "camera", CAMERA_COLOUR,
               f"tracker label: {cam_label} (its guess)   handedness vs glove: {hand_diag}"),
              ("Glove hand model (reconstructed)", "glove", GLOVE_COLOUR,
               f"glove label: {glove_label} (names the hand)"))
    titled = []
    for k, (title, key, colour, label) in enumerate(panels):
        palm = fig.add_subplot(gs[1 + 2 * k])
        side = fig.add_subplot(gs[2 + 2 * k])
        p = hands[key]
        draw_hand(palm, sign * p[:, 1], p[:, 0], colour, letters=True)
        draw_hand(side, -p[:, 2], p[:, 0], colour, letters=False)
        for a, lim in ((palm, yr), (side, zr)):
            a.set_xlim(*lim), a.set_ylim(*xr), a.set_aspect("equal")
            a.tick_params(labelsize=7, colors="#555555")
            a.grid(True, color="#e6e6e6", lw=0.6)
            a.set_axisbelow(True)
            for s in a.spines.values():
                s.set_color("#bbbbbb")
        palm.set_title("palm view", fontsize=9, color="#333333")
        side.set_title("side view", fontsize=9, color="#333333")
        palm.set_xlabel(("y" if sign > 0 else "-y") + " (mm), index side "
                        + ("left" if sign > 0 else "right"), fontsize=8)
        palm.set_ylabel("x (mm), wrist to middle knuckle", fontsize=8)
        side.set_xlabel("-z (mm), palm " + ("right" if hand == "left" else "left"), fontsize=8)
        scale_bar(palm, yr, xr)
        titled.append((palm, side, title, colour, label))
    fig.canvas.draw()                    # applies the equal aspect: the boxes' real places
    b = ax.get_position()
    fig.text((b.x0 + b.x1) / 2, 0.835, "Photo: raw IR image", ha="center", fontsize=13,
             fontweight="bold", color="#222222")
    for palm, side, title, colour, label in titled:
        b0, b1 = palm.get_position(), side.get_position()
        mid = (b0.x0 + b1.x1) / 2
        fig.text(mid, 0.835, title, ha="center", fontsize=13, fontweight="bold", color=colour)
        fig.text(mid, 0.802, label, ha="center", fontsize=9, color="#222222")

    fig.suptitle(
        f"{meta.get('pose')}   take {meta.get('take')}   {hand} hand   ({stem})\n"
        f"moment {local_iso(moment.t)}  [{moment.source}]   camera frame "
        f"{cam.dt * 1000:+.1f} ms, glove frame {glv.dt * 1000:+.1f} ms (arrival) from it;"
        f" stillness: {still}\n"
        f"{LABEL}; {TIMING_LABEL}\n"
        f"(mm, wrist = origin, each sensor's own palm frame): median "
        f"{summary['median_mm']:.1f} mm, largest {summary['max_mm']:.1f} mm "
        f"({summary['max_point']}) over points 1-20", fontsize=10.5, y=0.995, va="top")
    return fig


def draw_hand(ax, h, v, colour, letters: bool):
    """Bones, joints, wrist and (palm view) the finger letters past the tips."""
    pts = np.column_stack([h, v])
    for a, b in BONES:
        ax.plot(*pts[[a, b]].T, color=colour, lw=2.0, solid_capstyle="round", zorder=2)
    ax.scatter(pts[1:, 0], pts[1:, 1], s=16, color=colour, edgecolor="white", lw=0.8,
               zorder=3)
    ax.scatter([0], [0], s=70, marker="o", facecolor="white", edgecolor="#222222", lw=1.4,
               zorder=4)
    ax.annotate("wrist = origin", (0, 0), xytext=(0, -11), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, color="#222222")
    if not letters:
        return
    for finger, chain in CHAINS.items():
        tip, dip = pts[chain[3]], pts[chain[2]]
        d = tip - dip
        n = float(np.hypot(*d))
        at = tip + (d / n * 8.0 if n > 1e-6 else np.array([0.0, 8.0]))
        ax.text(*at, LETTERS[finger], ha="center", va="center", fontsize=9,
                fontweight="bold", color="#222222", zorder=5)


def scale_bar(ax, hr, vr):
    x0, y0 = hr[1] - 6.0 - 20.0, vr[0] + 6.0
    ax.plot([x0, x0 + 20.0], [y0, y0], color="#222222", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + 10.0, y0 + 2.5, "20 mm", ha="center", va="bottom", fontsize=7.5)


# --- the session --------------------------------------------------------------------
def take_list(session: Path, stems, include_rejected: bool):
    """[(stem, meta path or None)] in name order."""
    if stems:
        return [(s, find_meta(session, s)) for s in stems]
    folders = [session / "leap"] + ([session / "rejected" / "leap"] if include_rejected else [])
    found = []
    for folder in folders:
        for path in sorted(folder.glob("*.meta.json")):
            found.append((path.name[:-len(".meta.json")], path))
    return found


def write_comparison(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in COLUMNS})


WHOLE_FRAME_PX = (768, 768)      # the camera window's whole frame (scripts/leap/camera_view.py)


def run_summary(rows: list) -> str:
    """This run's handedness and stillness counts, smallest margins, largest |dt|."""
    done = [r for r in rows if r["status"] == OK]
    if not done:
        return "No take was processed in this run."
    hc, sc = Counter(r["handedness"] for r in done), Counter(r["stillness"] for r in done)
    g = min(done, key=lambda r: abs(r["glove_z_margin_mm"]))
    c = min(done, key=lambda r: abs(r["camera_z_margin_mm"]))
    dt = max(max(abs(r["cam_dt_ms"]), abs(r["glove_dt_ms"])) for r in done)
    return (f"In this run ({len(done)} processed takes): handedness "
            + ", ".join(f'"{k}" {n}' for k, n in sorted(hc.items()))
            + "; stillness " + ", ".join(f'"{k}" {n}' for k, n in sorted(sc.items()))
            + f"; smallest |margin| glove {abs(g['glove_z_margin_mm']):.1f} mm "
            f"({g['take']}), camera {abs(c['camera_z_margin_mm']):.1f} mm ({c['take']}); "
            f"largest |dt| {dt:.1f} ms.")


def readme_text(session: Path, rows: list, max_dt_s: float, include_rejected: bool) -> str:
    done = sum(r["status"] == OK for r in rows)
    whole = sum(r.get("_still_px") == WHOLE_FRAME_PX for r in rows)
    warn = (f"\n**{whole} of the stills here are the whole {WHOLE_FRAME_PX[0]} x "
            f"{WHOLE_FRAME_PX[1]} px camera frame**, saved before the camera window cropped "
            "stills to the hand: they show the operator, and the figures show them as saved. "
            "Do not pass those figures on as they are.\n" if whole else "")
    return f"""# Paired pose comparison: {session.name}

Made by `scripts/paired_pose_compare.py` on {datetime.date.today().isoformat()} from
`{session}`. {len(rows)} take(s) listed in `comparison.csv`, {done} processed,
{len(rows) - done} refused with the reason in its `status` column
({'accepted and rejected attempts' if include_rejected else 'accepted takes only'}).
`comparison.csv` lists the takes of the last run only.
{warn}
## What the numbers are

Every distance here is a **{LABEL}**, from an **{TIMING_LABEL}**. {SIMULTANEITY}
The camera's `capture_time` approximates when the image was captured; the glove's is when
its packet arrived; so |dt| within {max_dt_s * 1000:.0f} ms does not establish that both
sensors recorded the same physical instant, and the stillness check reduces that concern
but does not remove it.

Each sensor's palm frame is built from its own
landmarks, so differences in the hand's orientation between the sensors are removed, not
measured. The two sensors are not physically registered: no transform between the
camera's space and the glove's is known or used. This is the only common frame possible,
because the glove reports no hand position or orientation at all: its WRIST and PALM
joints never move (position and rotation identical in every line; measured 2026-09-16 on
the July 2026 glove JSONL, and re-checked 2026-11-03 on all 14,838 glove lines of
`recordings/sync_day2`). The unscaled distances are the result; no scaled table is
produced.

{run_summary(rows)}

## What each file is

Per take, in `<take>/`:

- `figure.png`: three panels. "Photo: raw IR image" is the still the camera window saved
  during the take, shown as it was saved. "Leap hand model (reconstructed)" and "Glove hand
  model (reconstructed)" are the
  21 points and 20 bones of each sensor's frame nearest the photo's moment, both drawn in the
  common frame below with the same limits and the same scale: a palm view (up = x, the
  fingers; across = y for a left hand and -y for a right hand, so both are seen from the
  palm) and a side view (up = x; across = -z, seen from the thumb side). T I M R L name
  the fingers past their tips, the open circle is the wrist (the origin), the bar is
  20 mm. Camera blue, glove orange.
- `camera.txt`, `glove.txt`: the same two frames in the professor's text format, written
  by the one exporter both sensors use (`scripts/glove/export_prof_format.py`), one block
  each, millimetres, each in its sensor's own native frame (the glove's wrist at 0,0,0;
  the camera's coordinates in the tracker's camera space, with its wrist on the `Wrist:`
  line). The block's `Time:` is that frame's own capture time.
- `points_common_frame.csv`: the 21 points of both sensors in the common frame
  (`glove_*_mm`, `camera_*_mm`) and the distance between them (`distance_mm`). Index 0
  is WRIST; 1 to 20 are the same points as 1 to 20 of the text files.
- `bone_lengths.csv`: the 20 bones of each sensor, mm.
- `notes.txt`: the result's label, the moment and where it came from, how far each
  sensor's frame is from it with the selected frames' identifiers and timestamps (the
  camera's `hand_id` and `visible_time_us` too), the clocks and the timing uncertainty,
  the labels each file carries, the stillness check, each sensor's three basis vectors
  and the transformation, the handedness diagnostic with both margins, the statistic
  basis ({STAT_BASIS}), what is not corrected.

Session level: `comparison.csv` (one row per take: `moment_source`, `cam_dt_ms`,
`glove_dt_ms`, `handedness`, `glove_z_margin_mm`, `camera_z_margin_mm`,
`glove_speed_mm_s`, `camera_speed_mm_s`, `stillness`, `median_mm` and `max_mm` ({STAT_BASIS}),
`max_point`, `n_points`, `status`), `figures.pdf` (every figure, one page per processed
take), this file.

## What each landmark is on each sensor

The two sensors' 21 points carry the same names in the same (MediaPipe) order, which does
not prove they are the same anatomical points. Both are model estimates: the glove's on a
template hand whose bone lengths are constant, the Leap's on a hand the tracker fits,
whose bone lengths change with the pose (`bone_lengths.csv`). Each point's source, from
`src/leap_hand/to_openxr.py` `JOINT_SOURCES` (Leap bone endpoint -> OpenXR joint) and
`src/xr_hand/keypoints21.py` `MP21_TO_OPENXR` (OpenXR joint -> 21-point name):

| 21-point name | Leap (camera) | Glove |
|---|---|---|
| WRIST (0) | `arm.next_joint`, the end of the forearm bone | OpenXR `WRIST` joint |
| THUMB_CMC (1) | `digits[0].bones[1].prev_joint`, the base of the thumb metacarpal (Leap's thumb `bones[0]` is zero-length) | OpenXR `THUMB_METACARPAL` of the template hand |
| THUMB_MCP, THUMB_IP (2, 3) | `digits[0].bones[2].prev_joint`, `bones[3].prev_joint` | `THUMB_PROXIMAL`, `THUMB_DISTAL` |
| finger MCP (5, 9, 13, 17) | `digits[d].bones[1].prev_joint` | `*_PROXIMAL` |
| finger PIP (6, 10, 14, 18) | `digits[d].bones[2].prev_joint` | `*_INTERMEDIATE` |
| finger DIP (7, 11, 15, 19) | `digits[d].bones[3].prev_joint` | `*_DISTAL` |
| TIP (4, 8, 12, 16, 20) | `digits[d].bones[3].next_joint` | `*_TIP` |

(The pinky is OpenXR `LITTLE_*`. Neither sensor's metacarpal bases of the four fingers
nor its PALM point is among the 21.)

## Conventions

- Units: millimetres everywhere.
- Origin: the wrist (point 0, WRIST) on both sensors.
- Axes: {axis_definition()}
- The moment: the photo's. From the take meta's `still_capture_time` (the camera window's
  record of when the still's IR image was captured, on the wall clock) when the take has
  it, labelled "{STILL_CAPTURE}"; otherwise the still file's modification time,
  labelled "{STILL_MTIME}", which is when the JPEG finished writing, a little after its
  image was captured. A take with neither is refused.
- Nearest the photo's moment: each sensor's frame is the one of its own file nearest it on the
  capture clock (when each sensor had the frame), on `wall_time` only when a file lacks
  capture times (named in notes.txt). A sensor with no frame within
  {max_dt_s * 1000:.0f} ms of the moment refuses the take. `cam_dt_ms` and `glove_dt_ms`
  are the frame's time minus the moment.
- Timing: both sensors' `capture_time` are on the PC wall clock (camera: `time.time()` at
  the LeapC callback minus the frame's age; glove: the packet's arrival at the OSC
  receiver), so the two dt values are comparable. But the glove's time is ARRIVAL, after
  the glove's transmission delay ({GLOVE_LAG['left']} for the left glove, {GLOVE_LAG['right']}
  for the right, measured 5 October 2026 on closing movements), so the glove frame's dt is
  to arrival, not to the hand. When the moment is "{STILL_CAPTURE}", it is the image
  frame's own LeapC timestamp converted to the wall clock the same way, so still and
  tracking frame are matched on the camera's clock; when it is the file write time, the
  unknown write delay adds to every dt. notes.txt carries a timing uncertainty line,
  separate from dt: camera one frame period (1000 / framerate ms, from the take's frames)
  plus the frame-age precision; glove the transmission delay above, not corrected.
- Stillness: for each sensor, the mean 21-point speed in mm/s over its frames within
  {STILL_WINDOW_S:g} s of the selected frame, on wrist-centred mm points (the hand's
  translation removed, its orientation and shape changes kept); each frame's speed over
  the {SPEED_STEP_S:g} s base `leap_hand.transition` uses. Above {STILL_MM_S:g} mm/s the
  sensor is "moving"; {STILL_MM_S:g} mm/s is `leap_hand.transition.POSE_SETTLE_SPEED_MM_S`,
  the speed below which the transition timing calls a pose settled (there it is measured
  with the hand's rotation also taken out, so this check is the stricter of the two).
  `stillness` is "still", "moving (glove)", "moving (camera)" or "moving (both)"; the take
  is processed either way.
- Point 0 of the text files, `0 (Palm)`, is a model point on both sensors (the glove: a
  fixed point of its template hand; the camera: the tracker's palm centre) and is not
  compared. The comparison uses the 21 MediaPipe-order points, wrist first.
- Statistics: {STAT_BASIS}. The wrist is the origin of both frames, so its distance is
  zero by construction. The middle knuckle's y and z are also zero on both by
  construction; its x is not.
- No scaling: the glove's bone lengths are those of a template hand and do not change;
  the tracker's bone lengths change with the pose. Both are reported as measured, and the
  unscaled distances are the result; no scaled table is produced.
- No lag correction: the glove's transmission delay above is not corrected. At the
  midpoint of a held pose it matters only if the hand was still moving (the stillness
  check).
- Labels: the glove's `hand_side` is which glove was worn and names the hand. The
  camera's label is the tracker's guess and is not used to name the hand.
- Handedness is a diagnostic, never a classification. The margin of each sensor is the
  mean palm-normal coordinate z of its points 1 to 20, in mm (`glove_z_margin_mm`,
  `camera_z_margin_mm`): fingers flex toward the palm and the thumb sits in front of it,
  so the points lie on the palm side of the palm plane, and the sign of z flips under a
  mirror in any plane. `handedness` is "{SAME}" when both margins have the same sign,
  "{OPPOSITE}" when they differ, "{INDETERMINATE}" when either is under
  {HANDEDNESS_MARGIN_MM:g} mm. The in-plane thumb side cannot detect a mirror: in this frame
  y runs from the index to the pinky knuckle on either hand, which puts the thumb on the
  index side (-y) for a left and a right hand alike. Evidence: on the day-2 rehearsal
  (`recordings/sync_day2`, 2026-11-03), all 60 takes (both hands, six poses) gave "{SAME}",
  smallest margin 5.3 mm (camera, open_palm_left_take5); this run's own counts and
  smallest margins are at the top. Weakest on a flat open hand, whose points lie near the
  palm plane. Nothing is reflected or corrected; the camera's points
  are as the tracker gave them.
- Wording: a number here is the distance between two sensors' estimates of the same
  point, a disagreement; neither sensor is treated as the reference. A sensor that does
  not measure something leaves it empty, it is never filled in from the other.
- Coordinates only: no joint angles are computed.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?", type=Path, help="a record_simultaneous.py session folder")
    ap.add_argument("--takes", nargs="+", default=None, help="only these take stems")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default results/paired_pose/<session name>)")
    ap.add_argument("--max-dt-ms", type=float, default=50.0,
                    help="farthest a sensor's frame may be from the moment (default 50)")
    ap.add_argument("--include-rejected", action="store_true",
                    help="also process accepted:false takes and rejected/ attempts")
    ap.add_argument("--check-outputs", type=Path, default=None, metavar="DIR",
                    help="re-read an output folder and check it is complete")
    args = ap.parse_args(argv)
    if args.check_outputs is not None:
        return check_outputs(args.check_outputs)
    if args.session is None or not (args.session / "leap").is_dir():
        print(f"not a session folder (no leap/ inside): {args.session}", file=sys.stderr)
        return 2
    session = args.session
    out = args.out or (REPO / "results" / "paired_pose" / session.resolve().name)
    out.mkdir(parents=True, exist_ok=True)
    max_dt_s = args.max_dt_ms / 1000.0
    exporter = glove_exporter()

    rows = []
    with PdfPages(out / "figures.pdf") as pdf:
        for stem, meta_path in take_list(session, args.takes, args.include_rejected):
            for name in TAKE_FILES:                  # nothing stale from an earlier run
                (out / stem / name).unlink(missing_ok=True)
            if meta_path is None:
                row, fig = {"take": stem, "status": "no meta.json for this take"}, None
            else:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("accepted") is not True and not args.include_rejected:
                    row = {"take": stem, "pose": meta.get("pose"), "hand": meta.get("hand"),
                           "status": "not accepted (meta accepted false); "
                                     "--include-rejected processes it"}
                    fig = None
                else:
                    row, fig = process_take(session, stem, meta, meta_path, out, max_dt_s,
                                            exporter)
            rows.append(row)
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
                print(f"{stem}: ok, median {row['median_mm']} mm, largest {row['max_mm']} mm "
                      f"({row['max_point']}), camera {row['cam_dt_ms']:+} ms, glove "
                      f"{row['glove_dt_ms']:+} ms, handedness {row['handedness']}, "
                      f"{row['stillness']}")
            else:
                print(f"{stem}: refused: {row['status']}")
    write_comparison(out / "comparison.csv", rows)
    (out / "README.md").write_text(readme_text(session, rows, max_dt_s, args.include_rejected),
                                   encoding="utf-8")
    if not args.include_rejected and not args.takes:
        n = len(list((session / "rejected" / "leap").glob("*.meta.json")))
        if n:
            print(f"{n} rejected attempt(s) under rejected/ not processed (--include-rejected)")
    done = sum(r["status"] == OK for r in rows)
    print(f"PAIRED POSE DONE: takes={len(rows)} processed={done} refused={len(rows) - done} "
          f"-> {out}")
    return 0


# --- the output check ----------------------------------------------------------------
README_MUST = ("millimetres", "wrist", "middle-finger knuckle", "PINKY_MCP", "INDEX_FINGER_MCP",
               "`0 (Palm)`", "template hand", "lag", "tracker's guess", "still capture time",
               STILL_MTIME, LABEL, TIMING_LABEL, SIMULTANEITY, "not physically registered", "WRIST and PALM",
               "x = (M - W) / |M - W|", "degenerate palm frame", "ARRIVAL",
               "What each landmark is on each sensor", "POSE_SETTLE_SPEED_MM_S",
               "no scaled table", SAME, OPPOSITE, INDETERMINATE)
NOTES_MUST = (LABEL, TIMING_LABEL, SIMULTANEITY, "timing uncertainty", "stillness:", "transformation:", "handedness:",
              STAT_BASIS)


def check_outputs(out: Path) -> int:
    """Re-read every output from disk; print OUTPUTS OK or what is wrong."""
    bad = []
    with open(out / "comparison.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or list(rows[0].keys()) != COLUMNS:
        bad.append("comparison.csv header or rows")
    done = [r for r in rows if r["status"] == OK]
    for r in done:
        d = out / r["take"]
        for name in TAKE_FILES:
            if not (d / name).is_file():
                bad.append(f"{r['take']}: no {name}")
        for name in ("camera.txt", "glove.txt"):
            if (d / name).is_file():
                blocks = list(parse_text((d / name).read_text(encoding="utf-8")))
                if len(blocks) != 1 or len(blocks[0].points) != 21:
                    bad.append(f"{r['take']}: {name} is not one 21-landmark block")
        for name, n in (("points_common_frame.csv", 21), ("bone_lengths.csv", 20)):
            if (d / name).is_file():
                with open(d / name, newline="", encoding="utf-8") as f:
                    if len(list(csv.DictReader(f))) != n:
                        bad.append(f"{r['take']}: {name} has not {n} rows")
    pages = len(re.findall(rb"/Type\s*/Page[^s]", (out / "figures.pdf").read_bytes()))
    if pages != len(done):
        bad.append(f"figures.pdf has {pages} pages for {len(done)} processed takes")
    readme = (out / "README.md").read_text(encoding="utf-8")
    bad += [f"README.md does not say {s!r}" for s in README_MUST if s not in readme]
    for r in done:
        path = out / r["take"] / "notes.txt"
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            bad += [f"{r['take']}: notes.txt does not say {s!r}" for s in NOTES_MUST
                    if s not in text]
        if r["handedness"] not in (SAME, OPPOSITE, INDETERMINATE):
            bad.append(f"{r['take']}: handedness {r['handedness']!r}")
        if not (r["stillness"] == "still" or r["stillness"].startswith(("moving (", "speed not"))):
            bad.append(f"{r['take']}: stillness {r['stillness']!r}")
    for path in [out / "README.md"] + [out / r["take"] / "notes.txt" for r in done]:
        if path.is_file() and re.search(r"\berror|accuracy", path.read_text(encoding="utf-8"),
                                        re.IGNORECASE):
            bad.append(f"{path.name} of {path.parent.name} uses error/accuracy wording")
    if bad:
        print("OUTPUTS NOT OK:\n  " + "\n  ".join(bad))
        return 1
    print(f"OUTPUTS OK: takes={len(rows)} processed={len(done)} pages={pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
