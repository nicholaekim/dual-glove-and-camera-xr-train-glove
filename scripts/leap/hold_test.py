"""Does the glove's curl DRIFT while a pose is held still? The camera is the
reference that does not creep.

Hold one pose for a minute. Both sensors are sampled the whole time, and the
summary shows, per finger, the glove's curl over the first and last five
seconds of the hold against the camera's over the same windows. A glove that
relaxes toward "open" while the camera stays flat is a glove drifting, and
that is a fault in the measurement rather than in the hand.

THE DAY-2 FAILURE THIS SCRIPT EXISTS TO STOP
  `hold_drift_right_fist_20260920_200335.csv` is sixty seconds of OPEN PALM
  recorded under the label `fist`. The glove reads 1.43 1.97 2.07 1.97 1.71 —
  its open-palm constants — from the first frame to the last, the camera
  agrees the hand is open, and the file is worthless. Nothing in the old tool
  ever looked at the SHAPE of the hand: it waited for the camera to track
  anything, beeped, and started a sixty-second clock.

  So this version will not start the clock until the CAMERA says the pose is
  there. `leap_hand.pose_check`'s finger bands do the judging — the same
  bands `scripts/check_take_labels.py` audits recordings with — and the match
  has to hold without a break for `--confirm` seconds. If the pose never
  appears within `--confirm-timeout`, the run is refused and says why, rather
  than producing another minute of the wrong hand.

  The glove is deliberately NOT consulted. It is the instrument under test:
  letting it confirm the pose would be letting it mark its own work, and the
  one failure mode being measured — the glove sitting on its open-palm value
  while the hand is closed — is exactly the case where it would say yes.

AND THE OPEN-PALM BASELINE
  Before the pose is called, both sensors are sampled over an open palm the
  ACQUIRE gate has already vouched for, and those medians go into the CSV and
  the summary. They are what make the drift readable afterwards: the glove's
  open-palm output is a CONSTANT (its rail, see `cam_hand.fusion`), so
  knowing it is what separates "the glove has not responded to the pose yet"
  from "the glove has responded and is now creeping". On
  `hold_drift_right_fist_20260918_190912.csv` those are 5.6 s apart, and
  reading the drift from the wrong one turns +0.29 into -0.90.

Usage (XR Trainer streaming, camera plugged in, glove on):
  python scripts/leap/hold_test.py --hand right --pose fist
  python scripts/leap/hold_test.py --hand left --pose peace --seconds 30
  python scripts/leap/hold_test.py --hand right --mock --mock-glove --seconds 5

Writes, into --out (default results/diagnostics/):
  hold_drift_<hand>_<pose>_<stamp>.csv          same columns as the scratch
                                                tool: source,t,<5 fingers>
  hold_drift_<hand>_<pose>_<stamp>.stream.csv   the glove's stream, second by
                                                second: frames, worst gap,
                                                packet-counter jumps
"""
import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

from leap_hand.diagnostics import (
    FINGERS,
    PoseHold,
    camera_pose_match,
    drift_lines,
    hold_drift,
    stream_summary_line,
)
from leap_hand.live import DiagnosticRig, beep, curl_text, median_curls
from leap_hand.pose_check import EXPECTED_FINGERS
from leap_hand.protocol import (
    DEFAULT_BAND,
    Hud,
    band_text,
    parse_band,
    pose_label,
    view_caption,
)
from leap_hand.stream import LeapUnavailable

POSES = tuple(EXPECTED_FINGERS)
# Rows in the CSV that are the open-palm baseline rather than the hold. They
# carry t = -1 so that a reader which filters `source == "glove"` — every
# reader the scratch tool had — sees exactly the columns and the rows it saw
# before, and one that wants the baseline can ask for it by name.
BASELINE_T = -1.0
BASELINE_SOURCES = ("open_baseline_glove", "open_baseline_camera")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Hold one pose; measure the glove's drift against the camera.")
    p.add_argument("--hand", choices=["left", "right"], required=True)
    p.add_argument("--pose", choices=POSES, default="fist",
                   help="the pose to hold (default fist). The camera has to "
                        "SEE it before the clock starts.")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="how long to hold it (default 60)")
    p.add_argument("--baseline", type=float, default=1.5,
                   help="seconds of open palm sampled before the pose is "
                        "called, for the baseline (default 1.5)")
    p.add_argument("--confirm", type=float, default=1.5,
                   help="how long the camera must show the pose WITHOUT A "
                        "BREAK before the clock starts (default 1.5)")
    p.add_argument("--confirm-timeout", type=float, default=30.0,
                   help="give up if the pose has not appeared in this long "
                        "(default 30)")
    p.add_argument("--acquire-timeout", type=float, default=60.0,
                   help="give up if no acquirable open hand in this long")
    p.add_argument("--band", default=None, metavar="LOW,HIGH",
                   help=f"palm height band in cm (default "
                        f"{DEFAULT_BAND[0]:g},{DEFAULT_BAND[1]:g})")
    p.add_argument("--skip", type=float, default=3.0,
                   help="seconds of the hold to drop before the baseline "
                        "window, for the hand still changing shape")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--mock", action="store_true",
                   help="synthetic camera: rehearse with no hardware")
    p.add_argument("--mock-glove", action="store_true",
                   help="synthetic glove: rehearse with no hardware")
    p.add_argument("--out", type=Path, default=Path("results") / "diagnostics",
                   help="where the CSVs go (default results/diagnostics)")
    return p.parse_args(argv)


def per_ten_seconds(glove, camera, seconds, skip, step=10.0):
    """The scratch tool's second table: both sensors' medians, per 10 s."""
    lines = ["", "per 10 s (glove / camera), thumb..pinky:"]
    s = float(skip)
    while s < seconds:
        g = median_curls([x.curls for x in glove if s <= x.t < s + step])
        c = median_curls([x.curls for x in camera if s <= x.t < s + step])
        if g is not None and c is not None:
            lines.append(f"  {int(s):3d}-{int(s + step):<3d}s glove "
                         + " ".join(f"{v:.2f}" for v in g)
                         + "   camera " + " ".join(f"{v:.2f}" for v in c))
        s += step
    return lines


def write_csv(path, glove, camera, open_glove, open_camera):
    """The scratch tool's columns exactly, plus two named baseline rows."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "t"] + list(FINGERS))
        for source, curls in zip(BASELINE_SOURCES, (open_glove, open_camera)):
            if curls is not None:
                w.writerow([source, BASELINE_T]
                           + [round(float(v), 5) for v in curls])
        for source, rows in (("glove", glove), ("camera", camera)):
            for s in rows:
                w.writerow([source, round(s.t, 4)]
                           + [round(float(v), 5) for v in s.curls])


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
    label = pose_label(args.pose, stay=False)
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
        """Print above the HUD line, leaving the HUD to reopen underneath."""
        hud.close()
        for line in lines:
            print(line, flush=True)

    try:
        # --- ACQUIRE: the expected hand, open, settled, well placed -------
        say(f"\n{args.hand.upper()} hand. Hold it OPEN over the camera, "
            f"{band_text(band)}, palm toward the lens.",
            f"Then, on the beep: {label}, held still for "
            f"{args.seconds:.0f} s.")
        try:
            rig.acquire(
                args.acquire_timeout,
                show=lambda line: hud.show(line, time.time()),
                caption=lambda missing: view_caption(
                    "ACQUIRE", args.pose, extra="; ".join(missing),
                    stay=False))
        except TimeoutError as e:
            hud.close()
            print(f"\nREFUSED: {e}")
            return 2

        # --- BASELINE: what an open palm reads on both sensors ------------
        say(f"      acquired. Keep the palm open and still for "
            f"{args.baseline:g} s (baseline)...")
        rig.caption(f"HOLD OPEN PALM   baseline {args.baseline:g}s")
        base_g, base_c = [], []
        t_end = time.time() + args.baseline
        while time.time() < t_end:
            g, c = rig.poll()
            base_g += [s.curls for s in g]
            base_c += [s.curls for s in c]
            hud.show(rig.hud("BASELINE", t_end - time.time()), time.time())
            time.sleep(0.004)
        open_glove = median_curls(base_g)
        open_camera = median_curls(base_c)
        if open_glove is None or open_camera is None:
            hud.close()
            print("\nREFUSED: one of the two sensors gave nothing over the "
                  "open palm, so there is no baseline to measure drift from.")
            return 2
        say(f"      open palm  glove  {curl_text(open_glove)}",
            f"      open palm  camera {curl_text(open_camera)}")

        # --- CONFIRM: the CAMERA has to show the pose ---------------------
        beep(1000, 200)
        say(f"      NOW: {label}. Hold it still — the clock starts when the "
            f"camera sees it.")
        held = PoseHold(args.confirm)
        why = "the camera has not seen that hand yet"
        t0 = time.time()
        while True:
            rig.poll()
            now = time.time()
            fresh = rig.fresh_camera(now)
            if fresh is None:
                matched, why = False, "the camera has lost that hand"
            else:
                matched, why = camera_pose_match(args.pose, fresh.curls,
                                                 fresh.gap)
            if held.update(now, matched):
                break
            left = args.confirm_timeout - (now - t0)
            hud.show(rig.hud("CONFIRM", left,
                             "seen, holding..." if matched else why), now)
            rig.caption(view_caption("SETTLE", args.pose,
                                     extra="" if matched else why,
                                     seconds_left=left, stay=False))
            if left <= 0:
                hud.close()
                print(f"\nREFUSED: the camera never saw a steady {label} in "
                      f"{args.confirm_timeout:g} s — {why}.")
                print("  Nothing was measured, on purpose: the day-2 "
                      "right-fist hold is 60 s of open palm recorded under "
                      "the label 'fist',")
                print("  and a file like that is worse than no file. Make the "
                      "pose clearly, palm toward the lens, and run it again.")
                return 2
            time.sleep(0.004)

        # --- HOLD ---------------------------------------------------------
        beep(1400, 150)
        say(f"      {label} confirmed by the camera. HOLD IT STILL for "
            f"{args.seconds:.0f} s. Do not relax it.")
        glove, camera = [], []
        start = time.time()
        lost = 0.0
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= args.seconds:
                break
            g, c = rig.poll()
            # Both sensors' stamps are on the shared wall clock (the OSC
            # arrival time and the camera's capture time), so subtracting the
            # same `start` from both leaves them comparable — which is the
            # whole reason the drift can be attributed to one of them.
            glove += [replace(s, t=s.t - start) for s in g]
            camera += [replace(s, t=s.t - start) for s in c]
            if rig.fresh_camera(now) is None:
                lost += 0.004
            left = args.seconds - elapsed
            hud.show(rig.hud("HOLD", left, f"HOLD: {label}"), now)
            rig.caption(view_caption("REC", args.pose, seconds_left=left,
                                     stay=False))
            time.sleep(0.004)
        beep(520, 300)
        rig.caption("done — see the terminal")
        hud.close()
    finally:
        rig.stop()

    # --- the numbers ------------------------------------------------------
    print()
    if len(glove) < 20 or len(camera) < 20:
        print(f"Not enough data (glove {len(glove)}, camera {len(camera)} "
              "frames). Nothing is reported from it.")
        return 2

    g_rows = hold_drift([s.t for s in glove], [s.curls for s in glove],
                        skip=args.skip, open_curls=open_glove)
    c_rows = hold_drift([s.t for s in camera], [s.curls for s in camera],
                        skip=args.skip, open_curls=open_camera)

    print(f"{label} held {args.seconds:.0f} s on the {args.hand} hand. "
          "Curl = tip-to-wrist over palm length; higher = straighter.")
    print(f"  open-palm baseline  glove  {curl_text(open_glove)}")
    print(f"  open-palm baseline  camera {curl_text(open_camera)}")
    stuck = max((r.stuck_s for r in g_rows), default=0.0)
    if stuck > 0.2:
        print(f"  the glove stayed on its OPEN-PALM value for {stuck:.1f} s "
              "after the camera already saw the pose;")
        print("  the baseline window below starts where it began answering, "
              "not at the beep.")
    print()
    for line in drift_lines(g_rows, c_rows):
        print(line)
    for line in per_ten_seconds(glove, camera, args.seconds, args.skip):
        print(line)
    if lost > 0.25:
        print(f"\n  the camera lost that hand for about {lost:.1f} s of the "
              "hold; its columns are that much thinner than the glove's.")

    summary = rig.stream.summary()
    print()
    print("  " + stream_summary_line(summary))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out / f"hold_drift_{args.hand}_{args.pose}_{stamp}.csv"
    write_csv(out, glove, camera, open_glove, open_camera)
    stream_out = out.with_name(out.stem + ".stream.csv")
    write_stream_csv(stream_out, rig.stream)
    print(f"\nsaved {out}")
    print(f"saved {stream_out}")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
