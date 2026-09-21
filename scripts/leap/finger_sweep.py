"""Bend a finger — or the whole hand — slowly, all the way down and back, and
see what the glove says about it, with the IR camera as the reference for what
the hand was actually doing.

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

...BUT ONLY WHEN THE CAMERA FOLLOWED THE FINGER
  Binning by the camera makes the camera the measuring axis, and an axis that
  did not move measures nothing. Today's `dead_zone_right_ring_
  20260920_212254.csv` is a full, well-paced, fully covered sweep in which
  the glove's ring swung 0.87 to 1.97 and the CAMERA's ring curl stayed
  inside half a unit, because a gloved ring folding on its own is a shape the
  tracker does not resolve. The curve came out non-monotonic and the tool
  printed a saturation point off it to two decimals regardless.

  So before anything is binned, `diagnostics.camera_range` asks how much
  camera curl that finger actually covered (5th to 95th percentile). Under
  `--min-camera-range` (0.5; a real full bend spans about 0.8 to 1.7) the run
  says so in one sentence, prints NO transfer curve and NO saturation point,
  saves the CSVs anyway and exits 2.

WHOLE-HAND MODE IS THE ANSWER TO THAT, AND THE PRIMARY DIAGNOSTIC
  `--finger all` (or `--whole-hand`) paces the same cycles on a FIST: close
  the whole hand slowly, open it slowly. Every finger moves through its full
  range, the camera resolves all of them, and one recording yields a transfer
  curve, a saturation point and a lag PER FINGER — each binned by that
  finger's OWN camera curl, each guarded by its own camera range. The thumb
  is analysed on the same terms and reported only if it behaves; it alone
  cannot fail the run.

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
  python scripts/leap/finger_sweep.py --hand right --finger all
  python scripts/leap/finger_sweep.py --hand right --finger ring
  python scripts/leap/finger_sweep.py --hand left --finger index --cycles 4
  python scripts/leap/finger_sweep.py --hand right --whole-hand --mock

Writes, into --out (default results/diagnostics/):
  dead_zone_<hand>_<finger>_<stamp>.csv         the per-frame record, LONG:
                                                one row per glove frame per
                                                finger, columns
                                                  t, finger, glove_curl,
                                                  camera_curl, glove_on_rail,
                                                  bend_fraction
                                                All five fingers are written
                                                in BOTH modes, so a
                                                single-finger run can be
                                                re-analysed for the others.
                                                `<finger>` is `all` in
                                                whole-hand mode.
  dead_zone_<hand>_<finger>_<stamp>.stream.csv  the glove's stream, second by
                                                second

  (The scratch tools' narrow format — one row per frame, with the camera
  column named `camera_curl_at_same_instant` — is what the files written
  before 2026-09-20 have. Same quantities, one finger, no `finger` column.)
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
    MIN_CAMERA_RANGE,
    PoseHold,
    analyse_sweep,
    camera_pose_match,
    camera_range_verdict,
    coverage_gaps,
    stream_summary_line,
)
from leap_hand.live import DiagnosticRig, beep, curl_text, median_curls
from leap_hand.protocol import DEFAULT_BAND, Hud, band_text, parse_band, view_caption
from leap_hand.stream import LeapUnavailable

# How close the glove's curl has to be to its open-palm value to count as
# sitting on its rail, for the CSV's `glove_on_rail` column. The same
# float-equality tolerance `cam_hand.fusion.RailOverrideParams` uses.
RAIL_TOL = 0.005

BEND = "bend"
STRAIGHTEN = "straighten"

# `--finger all`, the whole-hand protocol.
ALL = "all"
# The fingers a whole-hand run is JUDGED on. The thumb is analysed and
# reported like the rest, but it does not fail the run on its own: its camera
# band is 0.045 wide by measurement (`pose_check.PoseCheckParams`), it folds
# across the palm rather than into it, and a fist that closes all four fingers
# properly is a good fist whether or not the tracker resolved the thumb.
MEASURING = tuple(f for f in FINGERS if f != "thumb")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sweep one finger slowly; measure the glove against the camera.")
    p.add_argument("--hand", choices=["left", "right"], required=True)
    p.add_argument("--finger", choices=list(FINGERS) + [ALL], default="index",
                   help=f"which finger to sweep, or '{ALL}' for the "
                        "whole-hand fist protocol (default index)")
    p.add_argument("--whole-hand", action="store_true",
                   help=f"same as --finger {ALL}")
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
    p.add_argument("--min-camera-range", type=float, default=MIN_CAMERA_RANGE,
                   help=f"camera curl a finger must cover (5th to 95th "
                        f"percentile) before any transfer curve or saturation "
                        f"point is reported for it (default "
                        f"{MIN_CAMERA_RANGE:g}; a real full bend spans about "
                        f"0.8 to 1.7)")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--mock", action="store_true",
                   help="synthetic camera: rehearse with no hardware")
    p.add_argument("--mock-glove", action="store_true",
                   help="synthetic glove: rehearse with no hardware")
    p.add_argument("--out", type=Path, default=Path("results") / "diagnostics",
                   help="where the CSVs go (default results/diagnostics)")
    args = p.parse_args(argv)
    if args.whole_hand:
        args.finger = ALL
    return args


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


def write_csv(path, sweeps, open_glove, open_camera):
    """The per-frame record, LONG: one row per glove frame per finger.

    Five fingers are written whichever mode the run was in. An isolated ring
    sweep whose camera never followed the ring still holds, in the same file,
    what the camera and the glove were doing on the other four — which is the
    difference between "the tracker could not resolve this finger" and "the
    tracker had lost the hand", and it cannot be recovered later if only the
    swept column was kept.

      t              seconds from the first sample of the sweep
      finger         thumb | index | middle | ring | pinky
      glove_curl     the glove's curl on that finger
      camera_curl    the CAMERA's, at the instant that glove sample
                     describes, shifted by that finger's own measured lag
      glove_on_rail  1 where the glove is within RAIL_TOL of its open value
      bend_fraction  how far bent, 0 at the open palm and 1 at the camera's
                     2nd percentile. Kept for continuity with the scratch
                     tool; it depends on the open reference, which is the
                     dependency the binned curve exists to avoid.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "finger", "glove_curl", "camera_curl",
                    "glove_on_rail", "bend_fraction"])
        for s in sweeps:
            i = list(FINGERS).index(s.finger)
            cam_open = float(open_camera[i])
            cam_bent = (float(np.percentile(s.camera, 2)) if s.camera.size
                        else cam_open)
            span = max(cam_open - cam_bent, 1e-6)
            on_rail = np.abs(s.glove - float(open_glove[i])) <= RAIL_TOL
            bend = (cam_open - s.camera) / span
            for k in range(s.times.size):
                w.writerow([round(float(s.times[k]), 4), s.finger,
                            round(float(s.glove[k]), 5),
                            round(float(s.camera[k]), 5),
                            int(on_rail[k]), round(float(bend[k]), 4)])


def write_stream_csv(path, log):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["second", "frames", "max_gap_ms", "counter_jumps",
                    "lost_packets"])
        for row in log.rows():
            w.writerow([row.second, row.frames, row.max_gap_ms,
                        row.counter_jumps, row.lost_packets])


def phase_words(phase, finger, whole, seconds):
    """(the terminal line, the window caption) for one half of one cycle."""
    if whole:
        if phase == BEND:
            return (f"close the whole hand slowly into a FIST "
                    f"({seconds:g} s)", "CLOSE the whole hand into a FIST")
        return (f"open it slowly ({seconds:g} s)", "OPEN the hand")
    word = "BEND the" if phase == BEND else "STRAIGHTEN the"
    return (f"{word} {finger.upper()}, slowly",
            f"{word.upper()} {finger.upper()} SLOWLY")


def finger_lines(s, args, open_glove, open_camera):
    """Everything printed about one finger. Returns (lines, ok).

    `ok` is False when this finger's numbers are not a measurement: either
    the camera did not follow it, or the sweep left holes in the range it did
    cover. Both save the CSV and both fail the run, because the alternative —
    printing the table anyway and letting the reader notice — is what
    produced the 21:22 ring sweep.
    """
    i = list(FINGERS).index(s.finger)
    cam_open = float(open_camera[i])
    cam_bent = float(np.percentile(s.camera, 2))
    on_rail = np.abs(s.glove - float(open_glove[i])) <= RAIL_TOL
    # The glove's own range over the same percentiles the guard uses on the
    # camera, so the two numbers on that line are comparable. The whole
    # failure mode is a large one beside a small one.
    glove_span = float(np.percentile(s.glove, 95) - np.percentile(s.glove, 5))
    out = [f"  {s.finger.upper()}",
           f"    open palm   glove {float(open_glove[i]):.3f}   "
           f"camera {cam_open:.3f}",
           f"    fully bent  glove {float(np.percentile(s.glove, 2)):.3f}   "
           f"camera {cam_bent:.3f}",
           f"    the camera covered {s.span.span:.2f} of curl "
           f"({s.span.low:.2f} to {s.span.high:.2f}); the glove moved "
           f"{glove_span:.2f}",
           f"    the glove sat within {RAIL_TOL} of its open value for "
           f"{100.0 * on_rail.mean():.0f}% of the sweep",
           ""]

    if not s.measured:
        out += ["    " + line for line in camera_range_verdict(s.span)]
        if s.lag is not None:
            out.append(f"    (the lag that little motion implies is "
                       f"{s.lag.ms:+.0f} ms, match {s.lag.correlation:.2f} — "
                       f"the only number here, and no better than the range "
                       f"it came from)")
        return out, False

    out += ["  " + line for line in curve_lines(s.bins, args.bin_width)]
    out.append("")
    if s.saturation is None:
        out.append("    rail saturation: the glove never reached its "
                   "open-palm value again during the sweep,")
        out.append("      so there is no saturation point in this range.")
    else:
        out.append(f"    rail saturation: the glove is back on its open-palm "
                   f"value ({s.rail:.3f}) from camera curl "
                   f"{s.saturation:.2f} upward —")
        out.append("      above that the glove is reporting a constant, and "
                   "the camera is the only sensor still measuring.")
    if s.lag is None:
        out.append("    lag: not measurable — one of the two traces barely "
                   "moved.")
    else:
        out.append(f"    lag: the glove trails the camera by {s.lag.ms:+.0f} "
                   f"ms (match {s.lag.correlation:.2f});")
        out.append("      camera frames are about 10 ms old when we see "
                   f"them, so the glove's own delay is about "
                   f"{s.lag.ms + 10:.0f} ms.")

    gaps = coverage_gaps(s.camera, cam_bent, cam_open,
                         bin_width=args.bin_width, min_samples=args.min_bin)
    if gaps:
        out.append(f"    COVERAGE FAILED: {len(gaps)} bin(s) between the bent "
                   f"({cam_bent:.2f}) and open ({cam_open:.2f}) camera curl "
                   f"have fewer than {args.min_bin} samples:")
        for low, high, n in gaps:
            out.append(f"      camera curl {low:.2f}-{high:.2f}: "
                       f"{n} sample(s)")
        out.append("      The finger passed through those shapes too fast, "
                   "or never reached them. The CSV is saved anyway,")
        out.append("      but the transfer curve interpolates over those "
                   "bins, so repeat the sweep more slowly.")
        return out, False
    out.append(f"    coverage OK: every {args.bin_width:g}-wide bin from "
               f"{cam_bent:.2f} to {cam_open:.2f} has at least "
               f"{args.min_bin} samples.")
    return out, True


def run(args) -> int:
    band = parse_band(args.band) if args.band else DEFAULT_BAND
    finger = args.finger
    whole = finger == ALL
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
        what = ("WHOLE HAND" if whole else f"{finger} finger")
        plan = (f"Then {args.cycles} slow cycles: close the whole hand into a "
                f"FIST ({args.phase:g} s), open it ({args.phase:g} s)."
                if whole else
                f"Then {args.cycles} slow cycles: bend ONLY the {finger} all "
                f"the way down ({args.phase:g} s), straighten it "
                f"({args.phase:g} s).")
        say(f"\n{args.hand.upper()} hand, {what}. Hold the hand OPEN "
            f"over the camera, {band_text(band)}, palm toward the lens.",
            "Keep the other hand on your lap.", plan)
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
            "SLOWLY and steadily as you can;",
            "      keep the hand in the band and the palm toward the lens."
            if whole else
            "      keep the other fingers straight.")
        start = time.time()
        for cycle in range(args.cycles):
            for phase in (BEND, STRAIGHTEN):
                beep(1200 if phase == BEND else 800, 150)
                line, caption = phase_words(phase, finger, whole, args.phase)
                say(f"      cycle {cycle + 1}/{args.cycles}: {line}")
                t_end = time.time() + args.phase
                while time.time() < t_end:
                    now = time.time()
                    g, c = rig.poll()
                    # EVERY finger is recorded in both modes: a sweep kept as
                    # one column cannot be re-read for the others, and the
                    # other four are what say whether the camera had the hand
                    # at all while it was failing to follow the swept finger.
                    glove += [(s.t - start, s.curls) for s in g]
                    camera += [(s.t - start, s.curls) for s in c]
                    left = t_end - now
                    hud.show(rig.hud(phase.upper(), left, line), now)
                    rig.caption(
                        f"{caption} SLOWLY   {max(0.0, left):.0f}s   "
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
    gv = np.array([list(v) for _t, v in glove], dtype=float)
    ct = np.array([t for t, _v in camera], dtype=float)
    cv = np.array([list(v) for _t, v in camera], dtype=float)

    # Every finger is analysed so every finger can be written to the CSV; only
    # the ones this run was ABOUT are reported. `analyse_sweep` measures each
    # finger's lag separately and pairs its traces on that lag before binning.
    every = analyse_sweep(gt, gv, ct, cv, open_glove,
                          bin_width=args.bin_width,
                          min_range=args.min_camera_range)
    write_csv(out, every, open_glove, open_camera)
    reported = [s for s in every if whole or s.finger == finger]

    print(f"{'whole hand' if whole else finger + ' finger'}, {args.hand} "
          f"hand, {args.cycles} cycles over {total:g} s. Curl = tip-to-wrist "
          "over palm length; higher = straighter.")
    print()
    failed = []
    for s in reported:
        lines, ok = finger_lines(s, args, open_glove, open_camera)
        for line in lines:
            print(line)
        print()
        if not ok:
            failed.append(s)

    print("  " + stream_summary_line(rig.stream.summary()))
    print(f"\nsaved {out}")
    print(f"saved {stream_out}")

    # A whole-hand run is judged on the four measuring fingers: the thumb is
    # reported on the same terms but cannot fail the run on its own.
    fatal = [s for s in failed if not whole or s.finger in MEASURING]
    if fatal:
        names = ", ".join(s.finger for s in fatal)
        print(f"\nNOT A MEASUREMENT: {names} — see the reason(s) above. The "
              f"CSVs are saved; nothing else here is a number to quote.")
        if not whole and any(not s.measured for s in fatal):
            print("  An isolated finger the camera cannot resolve does not "
                  "become resolvable by sweeping it again. Run")
            print(f"    python scripts/leap/finger_sweep.py --hand "
                  f"{args.hand} --finger {ALL}")
            print("  instead: the whole-hand fist moves every finger through "
                  "its range, and the camera follows all of them.")
        return 2
    if failed:
        print(f"\nthe {', '.join(s.finger for s in failed)} is not reported "
              f"(see above); the measuring fingers are.")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
