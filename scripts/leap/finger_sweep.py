"""Bend ONE finger slowly, all the way down and back, and see what the glove
says about it — with the IR camera as the reference for what the finger was
actually doing.

The finding that prompted the first version: during a pinch, XR Trainer
reports the index exactly as straight as in an open palm while the camera
watches it fold down to meet the thumb. That is the glove's curl sitting on
its RAIL — the top of the stretch sensor's range, where the number has
stopped meaning anything — and `cam_hand.fusion`'s rail-disagreement override
exists because of it. Whether a given finger of a given glove does that, and
where, is a measurement, and this is the tool that makes it.

WHAT IS REPORTED, AND WHY IT IS BINNED BY THE **CAMERA**
  The transfer curve: the glove's median curl in each 0.1-wide slice of
  CAMERA curl, separately while bending and while straightening. Read off it:

    rail saturation   the camera curl above which the glove's answer stops
                      changing — the point past which it is reporting a
                      constant.
    hysteresis        any gap between the bending and straightening columns
                      in the same slice: the same finger shape read two
                      different ways depending on where it came from.
    lag               the shift that best aligns the two traces, by
                      cross-correlation.

  The old tool reported instead "the glove stayed on its rail until the
  finger was N % bent", a percentage measured from the finger's OPEN
  reading — and `dead_zone_right_ring_20260920_200435.csv` is a sweep that
  started with the ring already bent, so its open reference is wrong and
  every percentage computed from it is wrong with it. Binning by the camera
  needs no such reference: the camera measures the finger's shape frame by
  frame, so "what does the glove say when the camera says 1.3?" is
  answerable from any sweep that visited 1.3.

  Both halves of that lesson are built in. The run refuses to start until the
  camera confirms an OPEN PALM of the expected hand, so the open reference is
  real; and the analysis does not depend on it anyway.

THE PACED PROTOCOL
  A free-running "bend it slowly for 40 s" produced sweeps with whole
  stretches of the range missing, which a binned curve interpolates over
  without saying so. So the run is paced — `--cycles` cycles of BEND
  (`--phase` s) and STRAIGHTEN (`--phase` s), each called by a beep and a
  caption on the camera window — and at the end the coverage is CHECKED: at
  least `--min-bin` samples in every 0.1-wide camera-curl bin between the
  finger's open value and its fully bent value. A sweep that fails prints the
  bins it is missing and exits 2, with the CSV saved so the attempt can still
  be looked at.

Usage (XR Trainer streaming, camera plugged in, glove on, OTHER hand away):
  python scripts/leap/finger_sweep.py --hand right --finger ring
  python scripts/leap/finger_sweep.py --hand left --finger index --cycles 4
  python scripts/leap/finger_sweep.py --hand right --finger ring --mock --mock-glove

Writes, into --out (default results/diagnostics/):
  dead_zone_<hand>_<finger>_<stamp>.csv         same columns as the scratch
                                                tool: t, glove_curl,
                                                camera_curl_at_same_instant,
                                                glove_on_rail, bend_fraction
  dead_zone_<hand>_<finger>_<stamp>.stream.csv  the glove's stream, second by
                                                second
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from leap_hand.diagnostics import (
    BIN_WIDTH,
    FINGERS,
    MIN_BIN_SAMPLES,
    PoseHold,
    camera_pose_match,
    coverage_gaps,
    estimate_lag,
    rail_saturation,
    stream_summary_line,
    transfer_curve,
)
from leap_hand.live import DiagnosticRig, beep, curl_text, median_curls
from leap_hand.protocol import DEFAULT_BAND, Hud, band_text, parse_band, view_caption
from leap_hand.stream import LeapUnavailable

# How close the glove's curl has to be to its open-palm value to count as
# sitting on its rail, for the CSV's `glove_on_rail` column. The same
# float-equality tolerance `cam_hand.fusion.RailOverrideParams` uses.
RAIL_TOL = 0.005
# A sweep whose camera curl moved less than this never bent the finger enough
# to say anything about it, whatever the bin coverage says about the tiny
# range it did cover.
MIN_SWEEP_RANGE = 0.30

BEND = "bend"
STRAIGHTEN = "straighten"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sweep one finger slowly; measure the glove against the camera.")
    p.add_argument("--hand", choices=["left", "right"], required=True)
    p.add_argument("--finger", choices=list(FINGERS), default="index")
    p.add_argument("--cycles", type=int, default=3,
                   help="bend+straighten cycles (default 3)")
    p.add_argument("--phase", type=float, default=5.0,
                   help="seconds for each half of a cycle (default 5)")
    p.add_argument("--confirm", type=float, default=1.5,
                   help="how long the camera must show an OPEN PALM without a "
                        "break before the sweep starts (default 1.5)")
    p.add_argument("--confirm-timeout", type=float, default=30.0,
                   help="give up if no open palm has appeared in this long")
    p.add_argument("--acquire-timeout", type=float, default=60.0,
                   help="give up if no acquirable open hand in this long")
    p.add_argument("--band", default=None, metavar="LOW,HIGH",
                   help=f"palm height band in cm (default "
                        f"{DEFAULT_BAND[0]:g},{DEFAULT_BAND[1]:g})")
    p.add_argument("--bin", type=float, default=BIN_WIDTH, dest="bin_width",
                   help=f"camera-curl bin width (default {BIN_WIDTH})")
    p.add_argument("--min-bin", type=int, default=MIN_BIN_SAMPLES,
                   help=f"samples required in every bin of the swept range "
                        f"(default {MIN_BIN_SAMPLES})")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--mock", action="store_true",
                   help="synthetic camera: rehearse with no hardware")
    p.add_argument("--mock-glove", action="store_true",
                   help="synthetic glove: rehearse with no hardware")
    p.add_argument("--out", type=Path, default=Path("results") / "diagnostics",
                   help="where the CSVs go (default results/diagnostics)")
    return p.parse_args(argv)


def curve_lines(bins, bin_width):
    """The transfer curve as the table the report prints."""
    out = ["  the glove's curl as a function of the CAMERA's, per "
           f"{bin_width:g}-wide slice",
           f"  {'camera curl':<13} {'n':>5} {'glove':>7} {'bending':>8} "
           f"{'straight':>9} {'hysteresis':>11}"]
    for b in bins:
        def fmt(v):
            return "      -" if v is None else f"{v:7.3f}"

        hyst = ("          -" if b.hysteresis is None
                else f"{b.hysteresis:+11.3f}")
        out.append(f"  {b.low:5.2f}-{b.high:<6.2f} {b.n:5d} {fmt(b.glove)} "
                   f"{fmt(b.glove_bending):>8} {fmt(b.glove_straightening):>9} "
                   f"{hyst}")
    return out


def write_csv(path, t, glove, cam_at_glove, on_rail, bend_fraction):
    """The scratch tool's columns, exactly."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "glove_curl", "camera_curl_at_same_instant",
                    "glove_on_rail", "bend_fraction"])
        for i in range(len(t)):
            w.writerow([round(float(t[i]), 4), round(float(glove[i]), 5),
                        round(float(cam_at_glove[i]), 5), int(on_rail[i]),
                        round(float(bend_fraction[i]), 4)])


def write_stream_csv(path, log):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["second", "frames", "max_gap_ms", "counter_jumps",
                    "lost_packets"])
        for row in log.rows():
            w.writerow([row.second, row.frames, row.max_gap_ms,
                        row.counter_jumps, row.lost_packets])


def run(args) -> int:
    band = parse_band(args.band) if args.band else DEFAULT_BAND
    finger = args.finger
    fi = list(FINGERS).index(finger)
    total = args.cycles * 2 * args.phase
    rig = DiagnosticRig(hand=args.hand, band=band, view=not args.no_view,
                        mock_glove=args.mock_glove, mock_leap=args.mock)
    try:
        rig.start()
    except LeapUnavailable as e:
        print(f"\n{e}\n")
        return 2

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    hud = Hud(write=write)

    def say(*lines: str) -> None:
        hud.close()
        for line in lines:
            print(line, flush=True)

    glove, camera = [], []
    try:
        # --- ACQUIRE, then an OPEN PALM the camera actually agrees with ---
        say(f"\n{args.hand.upper()} hand, {finger} finger. Hold the hand OPEN "
            f"over the camera, {band_text(band)}, palm toward the lens.",
            "Keep the other hand on your lap.",
            f"Then {args.cycles} slow cycles: bend ONLY the {finger} all the "
            f"way down ({args.phase:g} s), straighten it ({args.phase:g} s).")
        try:
            rig.acquire(
                args.acquire_timeout,
                show=lambda line: hud.show(line, time.time()),
                caption=lambda missing: view_caption(
                    "ACQUIRE", "open_palm", extra="; ".join(missing),
                    stay=False))
        except TimeoutError as e:
            hud.close()
            print(f"\nREFUSED: {e}")
            return 2

        held = PoseHold(args.confirm)
        why = "the camera has not seen that hand yet"
        t0 = time.time()
        base_g, base_c = [], []
        while True:
            g, c = rig.poll()
            now = time.time()
            fresh = rig.fresh_camera(now)
            if fresh is None:
                matched, why = False, "the camera has lost that hand"
            else:
                matched, why = camera_pose_match("open_palm", fresh.curls,
                                                 fresh.gap)
            if matched:
                base_g += [s.curls for s in g]
                base_c += [s.curls for s in c]
            else:
                base_g, base_c = [], []       # a broken run is not a baseline
            if held.update(now, matched):
                break
            left = args.confirm_timeout - (now - t0)
            hud.show(rig.hud("OPEN", left,
                             "open palm seen, holding..." if matched else why),
                     now)
            rig.caption(view_caption("SETTLE", "open_palm",
                                     extra="" if matched else why,
                                     seconds_left=left))
            if left <= 0:
                hud.close()
                print(f"\nREFUSED: the camera never saw a steady OPEN PALM of "
                      f"the {args.hand} hand in {args.confirm_timeout:g} s — "
                      f"{why}.")
                print("  The sweep starts from a known open hand or it does "
                      "not start: the day-2 right-ring sweep began with the")
                print("  finger already bent, and every number measured "
                      "against its 'open' reference was wrong.")
                return 2
            time.sleep(0.004)

        open_glove = median_curls(base_g)
        open_camera = median_curls(base_c)
        if open_glove is None or open_camera is None:
            hud.close()
            print("\nREFUSED: one of the two sensors gave nothing over the "
                  "open palm.")
            return 2
        say(f"      open palm  glove  {curl_text(open_glove)}",
            f"      open palm  camera {curl_text(open_camera)}")

        # --- the paced sweep ---------------------------------------------
        say(f"      {args.cycles} cycles, {args.phase:g} s each way. Move as "
            "SLOWLY and steadily as you can; keep the other fingers straight.")
        start = time.time()
        for cycle in range(args.cycles):
            for phase in (BEND, STRAIGHTEN):
                beep(1200 if phase == BEND else 800, 150)
                word = ("BEND the" if phase == BEND
                        else "STRAIGHTEN the")
                say(f"      cycle {cycle + 1}/{args.cycles}: {word} "
                    f"{finger.upper()}, slowly")
                t_end = time.time() + args.phase
                while time.time() < t_end:
                    now = time.time()
                    g, c = rig.poll()
                    glove += [(s.t - start, s.curls[fi]) for s in g]
                    camera += [(s.t - start, s.curls[fi]) for s in c]
                    left = t_end - now
                    hud.show(rig.hud(phase.upper(), left,
                                     f"{word} {finger}"), now)
                    rig.caption(
                        f"{word.upper()} {finger.upper()} SLOWLY   "
                        f"{max(0.0, left):.0f}s   "
                        f"cycle {cycle + 1}/{args.cycles}")
                    time.sleep(0.004)
        beep(520, 300)
        rig.caption("done — see the terminal")
        hud.close()
    finally:
        rig.stop()

    # --- the numbers ------------------------------------------------------
    print()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out / f"dead_zone_{args.hand}_{finger}_{stamp}.csv"
    stream_out = out.with_name(out.stem + ".stream.csv")
    write_stream_csv(stream_out, rig.stream)

    if len(glove) < 50 or len(camera) < 50:
        print(f"Not enough data (glove {len(glove)}, camera {len(camera)} "
              f"frames over {total:g} s). Nothing is reported from it.")
        print(f"saved {stream_out}")
        return 2

    gt = np.array([t for t, _v in glove], dtype=float)
    gv = np.array([v for _t, v in glove], dtype=float)
    ct = np.array([t for t, _v in camera], dtype=float)
    cv = np.array([v for _t, v in camera], dtype=float)

    lag = estimate_lag(gt, gv, ct, cv)
    shift = 0.0 if lag is None else lag.seconds
    # The camera's curl at the instant each glove sample describes. Shifting
    # by the measured lag before pairing is what keeps the transfer curve
    # from smearing the glove's delay across the whole range.
    cam_at_glove = np.interp(gt - shift, ct, cv)

    rail = float(open_glove[fi])
    on_rail = np.abs(gv - rail) <= RAIL_TOL
    cam_open = float(open_camera[fi])
    cam_bent = float(np.percentile(cv, 2))
    span = cam_open - cam_bent
    bend_fraction = (cam_open - cam_at_glove) / max(span, 1e-6)
    write_csv(out, gt, gv, cam_at_glove, on_rail, bend_fraction)

    print(f"{finger} finger, {args.hand} hand, {args.cycles} cycles over "
          f"{total:g} s. Curl = tip-to-wrist over palm length; higher = "
          "straighter.")
    print(f"  open palm   glove {rail:.3f}   camera {cam_open:.3f}")
    print(f"  fully bent  glove {float(np.percentile(gv, 2)):.3f}   "
          f"camera {cam_bent:.3f}")
    print(f"  the glove sat within {RAIL_TOL} of its open value for "
          f"{100.0 * on_rail.mean():.0f}% of the sweep")
    print()

    bins = transfer_curve(cam_at_glove, gv, bin_width=args.bin_width)
    for line in curve_lines(bins, args.bin_width):
        print(line)

    saturation = rail_saturation(bins, rail)
    print()
    if saturation is None:
        print("  rail saturation: the glove never reached its open-palm value "
              "again during the sweep,")
        print("    so there is no saturation point in this range.")
    else:
        print(f"  rail saturation: the glove is back on its open-palm value "
              f"({rail:.3f}) from camera curl {saturation:.2f} upward —")
        print("    above that the glove is reporting a constant, and the "
              "camera is the only sensor still measuring.")
    if lag is None:
        print("  lag: not measurable — one of the two traces barely moved.")
    else:
        print(f"  lag: the glove trails the camera by {lag.ms:+.0f} ms "
              f"(match {lag.correlation:.2f});")
        print("    camera frames are about 10 ms old when we see them, so the "
              f"glove's own delay is about {lag.ms + 10:.0f} ms.")
    print()
    print("  " + stream_summary_line(rig.stream.summary()))
    print(f"\nsaved {out}")
    print(f"saved {stream_out}")

    # --- did the sweep actually cover the range? ----------------------------
    if span < MIN_SWEEP_RANGE:
        print(f"\nCOVERAGE FAILED: the camera saw the {finger} move only "
              f"{span:.2f} of curl (open {cam_open:.2f} to bent "
              f"{cam_bent:.2f}).")
        print("  That is not a sweep. Bend the finger all the way down and "
              "all the way back, slowly, and run it again.")
        return 2
    gaps = coverage_gaps(cam_at_glove, cam_bent, cam_open,
                         bin_width=args.bin_width, min_samples=args.min_bin)
    if gaps:
        print(f"\nCOVERAGE FAILED: {len(gaps)} bin(s) between the bent "
              f"({cam_bent:.2f}) and open ({cam_open:.2f}) camera curl have "
              f"fewer than {args.min_bin} samples:")
        for low, high, n in gaps:
            print(f"    camera curl {low:.2f}-{high:.2f}: {n} sample(s)")
        print("  The finger passed through those shapes too fast, or never "
              "reached them. The CSV above is saved anyway,")
        print("  but the transfer curve interpolates over those bins, so "
              "repeat the sweep more slowly before trusting it.")
        return 2
    print(f"\ncoverage OK: every {args.bin_width:g}-wide bin from "
          f"{cam_bent:.2f} to {cam_open:.2f} has at least {args.min_bin} "
          "samples.")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
