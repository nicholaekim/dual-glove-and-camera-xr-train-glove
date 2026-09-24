"""Fuse the glove and the Ultraleap LIVE, frame by frame, while you wear it.

`scripts/fuse_poses.py` fuses a session after it was recorded; this command
runs the same fusion on the two live streams (the StretchSense glove over OSC
on --port, the Ultraleap over LeapC) and sends each fused hand out as it is
made. Every piece that decides anything is the offline one, imported from
`cam_hand.fusion` and `cam_hand.template_fit` and run in `fuse_all`'s order
by `cam_hand.live_fusion.LiveFusion`; see that module for what differs live
(pairing against a short camera buffer, hand-id stability decided as frames
arrive, and learning from a warm-up instead of a whole session).

THE WARM-UP (one hand at a time, before anything is fused)
  With --hand both the LEFT hand goes first, then the RIGHT, and each hand
  learns from its own warm-up only. For each hand, follow the beeps, the
  console and the camera window's caption, which all name the hand:

    1. ACQUIRE      hold that hand open over the module, 18 to 28 cm up,
                    palm to the lens. Nothing starts until that hand's glove
                    is streaming (10 packets in the last second) and the
                    camera has tracked that hand for half a second inside
                    the height band, palm to the lens, over the module: the
                    recorder's own ACQUIRE gate. The HUD line says what is
                    still missing, e.g. "ACQUIRE LEFT  glove ok  camera: no
                    LEFT hand (raise it to 18 to 28 cm above the module)".
                    Up to --acquire-timeout seconds (default 60).
    2. a beep, then "open palm in 2..1"
    3. OPEN PALM flat to the camera   --warmup-open seconds (default 4)
    4. FIST                           --warmup-fist seconds (default 4)

  Each phase starts with a beep. From those frames it learns, per hand: the
  glove's rails (its reading for each straight finger), both sensors' open
  and flexed curl endpoints, and, with --fit-template auto, the operator's
  bone lengths from the open-palm camera frames (refused, with the reason
  printed, when too few open frames were trusted; that hand is then fused on
  the raw template). It prints what it learned.

  A hand is REFUSED, with every cause and what to do about it, when its
  ACQUIRE gate timed out, when the camera did not see its open palm, or when
  the fist did not close it on the glove or on the camera. A refused hand is
  not fused; with --hand both the other hand still is. When every hand is
  refused the program stops with exit code 2.

THE OUTPUTS, one per glove frame of each fused hand
  --out PATH.jsonl      one JSON line per fused frame: t_glove, t_cam (None
                        when no camera frame was within --max-dt), hand, the
                        21 fused points (wrist-centred metres, MediaPipe-21
                        order), dof_source per gated DOF, camera_used, the
                        rail override's fingers and the drift anchor's
                        per-finger curl corrections. Beside it:
    PATH.warmup.txt     everything the console said up to the fusion (the
                        settings, each hand's acquire outcome, what the
                        warm-up learned, every refusal and its causes), under
                        a first line with the date, hands, profile, lag and
                        its source and the template fit mode. Written the
                        moment the warm-up ends, before anything is fused, so
                        it is there even if the run is killed.
    PATH.summary.txt    the end-of-run summary as printed, under the run's
                        wall-clock start, fusion start and end
  --osc-out HOST:PORT   /fused/<hand>/keypoints21 (63 floats, the same
                        points) and /fused/<hand>/sources (one "dof=source"
                        string per gated DOF), per frame
  the terminal          a HUD line every 0.25 s: fused curls, spread and
                        thumb source, override fingers, anchor corrections,
                        glove rate, camera fresh or stale; and at the end a
                        summary of camera use per DOF, rejection reasons and
                        what the drift anchor learned

THE GLOVE LAG
  The solved glove hand trails the camera, by about 0.10 s on the left glove
  and 0.47 s on the right on this laptop, so each glove frame is paired with
  the camera frame from that much earlier. The lag applied is the reliability
  profile's measured `glove_lag_s` for that hand unless --glove-lag gives one;
  with no profile lag for a hand nothing is applied, and the startup lines say
  so. A live session has no settle clips to measure a lag from.

THE DRIFT ANCHOR is EXPERIMENTAL and off by default (--drift-anchor on). On,
it also forgets what it learned whenever a hand's camera has been stale for
more than a second, because the hand can come back in any pose.

A WARM-UP IS AN INITIAL CALIBRATION, not the offline learning. fuse_poses.py
learns its rails and endpoints label-free over a whole session of poses;
eight seconds of open palm and fist per hand is a much smaller sample. Before trusting live
output: run --replay on a recorded session (below), repeat the warm-up
across glove don/doff and across days and compare what it prints, and check
the lag's sign with one deliberate open-to-fist movement (the HUD's glove
curls must follow the camera by about the profile's lag, not lead it).

REPLAY (--replay DIR, no hardware, no warm-up)
  Runs every take of a recorded session (DIR/glove + DIR/leap, as written by
  record_simultaneous.py) through the live path at full speed, with the
  rails, endpoints, curl gates, template fit and glove lag learned from the
  session exactly as fuse_poses.py learns them, fuses the same session with
  fuse_poses.fuse_all, and prints per take: frames, paired frames, the
  largest fused-point difference and the dof_source mismatches. Everything
  must agree (to 1e-9 m); this is the check to run before a live session.
  --glove-lag auto here means what it means in fuse_poses.py (measured from
  the session, the profile as fallback); --hand is ignored.

Exit codes: 0 normal (Ctrl-C or --seconds ends a normal run; a replay that
agrees), 1 a sensor or argument problem, 2 the warm-up refused every hand,
3 a replay that disagrees with fuse_all.

Usage:
  python scripts/fuse_live.py                               # both hands
  python scripts/fuse_live.py --hand right --out runs/live_right.jsonl
  python scripts/fuse_live.py --osc-out 127.0.0.1:9010      # to a renderer
  python scripts/fuse_live.py --fit-template recordings/sync/template_right.json
  python scripts/fuse_live.py --mock-glove --mock-leap --no-view --seconds 10
  python scripts/fuse_live.py --replay recordings/sync_day2
"""
import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(HERE.parent / "src"))
# The profile and lag parsing live in the offline script and are imported
# from it rather than copied, so `--profile` and `--glove-lag` mean exactly
# the same thing in both commands. Run as a script, this folder is already
# on the path; imported (the tests), it is appended rather than put first, so
# nothing in it can shadow an installed package.
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

import fuse_poses  # noqa: E402
from fuse_poses import (  # noqa: E402
    FIT_AUTO,
    FIT_FILE,
    FIT_NONE,
    LAG_AUTO,
    LAG_NONE,
    PROFILE_AUTO,
    load_profile,
    parse_glove_lag,
    parse_rail_fingers,
    resolve_profile,
)

from cam_hand.features import flexion_features  # noqa: E402
from cam_hand.fusion import (  # noqa: E402
    DEFAULT_ANCHOR,
    DEFAULT_GATES,
    DEFAULT_RAIL,
    DriftAnchor,
    DriftAnchorParams,
    RailOverrideParams,
    flag_hand_id_stability,
    pairing_clock,
)
from cam_hand.live_fusion import (  # noqa: E402
    ACQUIRE_TIMEOUT_S,
    COUNTDOWN_S,
    MIN_SCORE,
    MOCK_POSE,
    STAGE_ACQUIRE,
    STAGE_COUNTDOWN,
    STAGE_NOT_ACQUIRED,
    WARMUP_FIST_S,
    WARMUP_OPEN_S,
    CameraBuffer,
    CameraSource,
    GloveSource,
    JsonlSink,
    LiveFusion,
    OscSink,
    RunLog,
    Warmup,
    acquire_status,
    band_words,
    camera_fresh,
    countdown_text,
    hud_line,
    hud_segment,
    learn_from_session,
    replay_take,
)
from cam_hand.recorder import take_files  # noqa: E402
from cam_hand.template_fit import load_measurement, measure_hand  # noqa: E402
from leap_hand.live import beep  # noqa: E402
from leap_hand.protocol import (  # noqa: E402
    DEFAULT_BAND,
    HUD_EVERY,
    AsyncBeeper,
    CameraView,
    Hud,
)

HAND_CHOICES = ("left", "right", "both")
# A phase cue: the high note says "change pose now", as in the coached
# recording protocol. The lower note says "hand acquired, get ready".
CUE_FREQ = 1320
ACQUIRED_FREQ = 880
CUE_MS = 180


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 1)[1])
    p.add_argument("--hand", choices=HAND_CHOICES, default="both",
                   help="which hand(s) to fuse (default both)")
    p.add_argument("--profile", default=PROFILE_AUTO,
                   metavar="{auto,none,PATH.json}",
                   help="reliability profile, as in fuse_poses.py: per-hand "
                        "unreliable fingers, rail-override fingers and "
                        "glove_lag_s. Default auto (profiles/default.json, "
                        "else the NK profile)")
    p.add_argument("--glove-lag", default=LAG_AUTO,
                   metavar="{auto,none,left:SEC[,right:SEC]}",
                   help="seconds the glove trails the camera, per hand. "
                        f"Default {LAG_AUTO}: the profile's glove_lag_s for "
                        "that hand, else nothing (and the startup lines say "
                        f"so); '{LAG_NONE}' applies nothing")
    p.add_argument("--fit-template", default=FIT_AUTO,
                   metavar="{auto,none,PATH.json}",
                   help="rescale the glove's template bones to the operator's. "
                        f"Default {FIT_AUTO}: measured per hand on the warm-up's "
                        "open-palm camera frames. PATH: a saved measurement "
                        "(e.g. <session>/template_right.json from fuse_poses)")
    p.add_argument("--warmup-open", type=float, default=WARMUP_OPEN_S,
                   help="seconds of each hand's OPEN PALM warm-up phase "
                        f"(default {WARMUP_OPEN_S:g})")
    p.add_argument("--warmup-fist", type=float, default=WARMUP_FIST_S,
                   help="seconds of each hand's FIST warm-up phase "
                        f"(default {WARMUP_FIST_S:g})")
    p.add_argument("--acquire-timeout", type=float, default=ACQUIRE_TIMEOUT_S,
                   help="seconds the ACQUIRE gate waits for each hand before "
                        f"refusing it (default {ACQUIRE_TIMEOUT_S:g})")
    p.add_argument("--max-dt", type=float, default=0.05,
                   help="max seconds between a glove frame (after its lag) "
                        "and the camera frame it is paired with (default 0.05)")
    p.add_argument("--drift-anchor", choices=("on", "off"), default="off",
                   help="EXPERIMENTAL: learn the glove's slow creep from the "
                        "camera on trusted frames and take it out; reset "
                        "whenever a hand's camera is stale for over 1 s "
                        "(default off)")
    p.add_argument("--anchor-window", type=float,
                   default=DEFAULT_ANCHOR.window_s,
                   help="seconds of trusted frames the drift offset is the "
                        f"median over (default {DEFAULT_ANCHOR.window_s})")
    p.add_argument("--anchor-hold", type=float, default=DEFAULT_ANCHOR.hold_s,
                   help="seconds after its last trusted frame a learned "
                        f"offset is still applied (default "
                        f"{DEFAULT_ANCHOR.hold_s})")
    p.add_argument("--anchor-deadband", type=float,
                   default=DEFAULT_ANCHOR.deadband,
                   help="flexion-fraction offset below which nothing is "
                        f"corrected (default {DEFAULT_ANCHOR.deadband})")
    p.add_argument("--no-rail-override", action="store_true",
                   help="never let the camera take a finger's curl")
    p.add_argument("--rail-fingers", default=None,
                   help="comma-separated fingers the rail override may act "
                        "on, both hands; overrides the profile (default "
                        f"{','.join(DEFAULT_RAIL.fingers_for())})")
    p.add_argument("--out", type=Path, default=None, metavar="PATH.jsonl",
                   help="write one JSON line per fused frame")
    p.add_argument("--osc-out", default=None, metavar="HOST:PORT",
                   help="send each fused frame as OSC to HOST:PORT")
    p.add_argument("--port", type=int, default=9002,
                   help="OSC port the glove streams to (default 9002)")
    p.add_argument("--mock-glove", action="store_true",
                   help="a synthetic 60 Hz glove per hand, no hardware")
    p.add_argument("--mock-leap", action="store_true",
                   help="the synthetic Leap stream, no camera or SDK")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--seconds", type=float, default=None,
                   help="stop after this many seconds of fusion (the "
                        "warm-up is not counted); default: until Ctrl-C")
    p.add_argument("--replay", type=Path, default=None, metavar="DIR",
                   help="no hardware: run a recorded session (DIR/glove, "
                        "DIR/leap) through the live path and print its "
                        "agreement with fuse_poses.fuse_all, per take")
    return p


def parse_host_port(text: str):
    """`'127.0.0.1:9010'` -> `('127.0.0.1', 9010)`."""
    host, sep, port = str(text).strip().rpartition(":")
    if not sep or not host:
        raise SystemExit(f"--osc-out {text}: expected HOST:PORT, "
                         "e.g. 127.0.0.1:9010")
    try:
        return host, int(port)
    except ValueError:
        raise SystemExit(f"--osc-out {text}: {port!r} is not a port number")


def resolve_lag(spec, hands, profile_lag, profile_path):
    """Per hand: (seconds applied, where that number came from).

    `auto` means the profile's measured value, because a live session has no
    settle clips to measure one from; a hand the profile says nothing about
    gets NO lag rather than a guess, and the reason is printed.
    """
    out = {}
    for hand in hands:
        if spec == LAG_NONE:
            out[hand] = (0.0, "none (--glove-lag none)")
        elif spec == LAG_AUTO:
            if hand in profile_lag:
                out[hand] = (float(profile_lag[hand]),
                             f"profile {profile_path.name} glove_lag_s")
            elif profile_path is None:
                out[hand] = (0.0, "auto, but no profile is in force: "
                                  "nothing applied")
            else:
                out[hand] = (0.0, f"auto, but {profile_path.name} has no "
                                  f"glove_lag_s for {hand}: nothing applied")
        elif hand in spec:
            out[hand] = (float(spec[hand]), "--glove-lag")
        else:
            out[hand] = (0.0, f"--glove-lag names no {hand} lag: "
                              "nothing applied")
    return out


def save_measurements(learned, folder) -> list:
    """Write each fitted hand's measurement as `template_<hand>.json` in
    `folder`, the name and writer `fuse_poses.fit_measurements` uses, so a
    later run (live or offline) can pass it with --fit-template PATH."""
    return [m.save(Path(folder) / FIT_FILE.format(hand=hand))
            for hand, m in sorted(learned.measurements.items())]


# --- what the live run says, shared with scripts/demo.py --live ---------------

def run_header(command, hands, profile_path, profile_how, lags,
               fit_spec) -> str:
    """The warm-up file's first line (after the date): the command, its
    hands, the profile, each hand's lag with where it came from, and the
    template fit mode."""
    lag = "; ".join(f"{h} {lags[h][0]:.3f} s ({lags[h][1]})" for h in hands)
    return (f"{command}  hands {', '.join(hands)}  profile "
            f"{profile_path or '(none)'} [{profile_how}]  glove lag {lag}  "
            f"template fit {fit_spec}")


def acquired_line(hand, seconds) -> str:
    """The kept line for a hand that passed its ACQUIRE gate (the HUD's
    countdown that follows is rewritten in place and not kept)."""
    return f"{hand.upper()} hand: acquired after {seconds:.1f} s."


def conclude_warmup(learned, hands, fit, out, log):
    """Say what the warm-up learned and decided, save each fitted hand's
    bone measurement beside `out`, and write PATH.warmup.txt.

    Returns (learned, the hands to fuse), `learned` narrowed to those hands.
    No hand to fuse means the caller shuts down and exits with code 2.
    """
    for line in learned.lines():
        log.say(line)
    if learned.fit_refusals and fit == FIT_AUTO:
        log.say("  (a longer --warmup-open gives the camera more open-palm "
                "frames to measure the bones on)")
    fused_hands = tuple(h for h in hands if h not in learned.refused)
    if not fused_hands:
        log.say("No hand passed the warm-up, so nothing was fused. Do what "
                "is listed above and start again.")
        log.write_warmup()
        return learned, fused_hands
    if learned.refused:
        refused = [h.upper() for h in hands if h in learned.refused]
        log.say(f"Fusing the {' and '.join(h.upper() for h in fused_hands)} "
                f"hand only: the {' and '.join(refused)} hand was refused "
                "(above).")
        learned = replace(learned, hands=fused_hands)
    if fit == FIT_AUTO and out is not None:
        for path in save_measurements(learned, out.parent):
            log.say(f"Saved the warm-up's bone measurement to {path} "
                    "(reuse it with --fit-template PATH)")
    log.write_warmup()
    return learned, fused_hands


def report_run(log, fusion, sinks, more=()) -> None:
    """Say the end-of-run summary, what each output received and `more`,
    then write PATH.summary.txt."""
    log.say()
    for line in fusion.summary_lines():
        log.say(line)
    for sink in sinks:
        if isinstance(sink, JsonlSink):
            log.say(f"Wrote {sink.written} fused frames to {sink.path}")
        elif isinstance(sink, OscSink):
            log.say(f"Sent {sink.sent} fused frames to "
                    f"{sink.host}:{sink.port}")
    for line in more:
        log.say(line)
    log.write_summary()


# --- --replay ------------------------------------------------------------------

@contextmanager
def listening_to_fuse_all(module):
    """Record what `fuse_all` fuses, frame by frame, without changing it.

    `fuse_all` keeps per-DOF tables, not the fused points, so for the
    duration of the block three names in `module` are wrapped to log, in
    call order:

      glove_pts             which glove ROW is being fused: `fuse_all` asks
                            it for the points to fuse, row by row, right
                            before each `fuse_skeletons` call
      fuse_skeletons        ("fuse", row id, points, dof_source, paired)
      DriftAnchor.reset     ("reset",)

    Frames are keyed by the glove row they came from, not by position, so
    the log does not depend on the order `fuse_all` pairs and fuses its
    takes in. The real functions do all the work and are put back on exit.
    """
    events = []
    current = [None]
    real_glove_pts = module.glove_pts
    real_fuse = module.fuse_skeletons
    real_anchor = module.DriftAnchor

    def glove_pts(row, *a, **k):
        current[0] = id(row)
        return real_glove_pts(row, *a, **k)

    def fuse(glove_pts_, cam_pts, *a, **k):
        fused, info = real_fuse(glove_pts_, cam_pts, *a, **k)
        events.append(("fuse", current[0], np.array(fused, float),
                       dict(info["dof_source"]), cam_pts is not None))
        return fused, info

    class ListenedAnchor(real_anchor):
        def reset(self, *a, **k):
            events.append(("reset",))
            return super().reset(*a, **k)

    module.glove_pts = glove_pts
    module.fuse_skeletons = fuse
    module.DriftAnchor = ListenedAnchor
    try:
        yield events
    finally:
        module.glove_pts = real_glove_pts
        module.fuse_skeletons = real_fuse
        module.DriftAnchor = real_anchor


def offline_by_row(events, loaded):
    """The listened events -> (frames, take order, resets).

    frames      {row id: (points, dof_source, paired)}, one per glove row
    take order  the `loaded` indices in the order `fuse_all` fused them
    resets      {take index: was the anchor reset before its first frame}
    """
    take_of = {id(row): k for k, entry in enumerate(loaded)
               for row in entry["glove"]}
    frames, order, reset_before = {}, [], {}
    pending_reset = False
    for ev in events:
        if ev[0] == "reset":
            pending_reset = True
            continue
        row_id = ev[1]
        k = take_of.get(row_id)
        if k is not None and k not in reset_before:
            reset_before[k] = pending_reset
            order.append(k)
        pending_reset = False
        frames[row_id] = ev[2:]
    return frames, order, reset_before


def compare_take(entry, offline, live):
    """(frames, paired offline, paired live, max point difference in metres,
    mismatches) for one take, matched glove row by glove row. A mismatch is
    a row fused on one side only, a row paired on one side and not the
    other, or a paired row whose dof_source differs."""
    rows = entry["glove"]
    live_by_row = {id(row): frame for row, frame in live}
    paired_off = sum(1 for row in rows
                     if id(row) in offline and offline[id(row)][2])
    paired_live = sum(1 for f in live_by_row.values() if f.t_cam is not None)
    worst = 0.0
    mismatches = 0
    for row in rows:
        off = offline.get(id(row))
        frame = live_by_row.get(id(row))
        if off is None or frame is None:
            mismatches += int((off is None) != (frame is None))
            continue
        pts, sources, paired = off
        worst = max(worst, float(np.max(np.abs(
            pts - np.asarray(frame.pts, float)))))
        if paired != (frame.t_cam is not None):
            mismatches += 1
        elif paired and sources != frame.dof_source:
            mismatches += 1
    return len(rows), paired_off, paired_live, worst, mismatches


REPLAY_TOL_M = 1e-9


def replay_session(input_dir, gates, rail_params, unreliable, profile_lag,
                   anchor, fit_spec, lag_text, max_dt) -> int:
    """`--replay`: the session through `fuse_all` and through the live path.

    Loading, hand-id flagging, per-take bone measurement, the lag and the
    template fit are `fuse_poses.main`'s own pass 1, with the same functions
    in the same order; the fit's `template_<hand>.json` is written to a
    temporary folder rather than into the session. The live path gets what
    `fuse_all` learned (`learn_from_session`, checked against the run's own
    rails, endpoints and curl gates), one fresh `LiveFusion` per take as
    `fuse_all` has one fresh rail tracker per take, and with the anchor on,
    ONE `DriftAnchor` in the order `fuse_all` fused the takes, reset wherever
    `fuse_all` reset its own.
    """
    fp = fuse_poses
    input_dir = Path(input_dir)
    glove_dir = input_dir / "glove"
    cam_dirs = [input_dir / d for d in fp.CAM_DIRS if (input_dir / d).is_dir()]
    if not glove_dir.is_dir() or not cam_dirs:
        print(f"--replay {input_dir}: expected {glove_dir} and "
              f"{input_dir / fp.LEAP}")
        return 1
    loaded, skipped = [], []
    measure_parts = defaultdict(list)
    for gpath in take_files(glove_dir):
        try:
            cpath = fp.find_camera_take(input_dir, gpath.name)
        except fp.AmbiguousTake as e:
            print(e)
            return 1
        if cpath is None:
            skipped.append((gpath.name, "no matching camera file"))
            continue
        glove = fp.load_glove(gpath)
        cam, source = fp.load_cam(cpath)
        if not glove or not cam:
            skipped.append((gpath.name, "one side is empty"))
            continue
        if source != fp.LEAP:
            skipped.append((gpath.name, "MediaPipe take: live fusion is "
                                        "Ultraleap only"))
            continue
        clock = pairing_clock(glove, cam)
        flag_hand_id_stability(cam, gates, clock=clock)
        if fit_spec.lower() == FIT_AUTO:
            for side in sorted({r["hand_side"] for r in cam}):
                got = measure_hand(cam, hand=side, gates=gates,
                                   source=gpath.name)
                if got is not None:
                    measure_parts[side].append(got)
        for r in cam:
            r.pop("abs26", None)
        loaded.append({"name": gpath.name, "glove": glove, "cam": cam,
                       "source": source, "clock": clock, "with_scale": False})
    if not loaded:
        print(f"--replay {input_dir}: no Ultraleap take to replay")
        return 1

    lag_rows, glove_lag = fp.glove_lag_of(parse_glove_lag(lag_text),
                                          input_dir, cam_dirs, loaded,
                                          profile_lag=profile_lag)
    with tempfile.TemporaryDirectory() as scratch:
        measurements, _scales, _saved, refusals = fp.fit_measurements(
            fit_spec, measure_parts, loaded, Path(scratch))

    with listening_to_fuse_all(fp) as events:
        run = fp.fuse_all(loaded, gates, rail_params, unreliable,
                          max_dt=max_dt, min_score=MIN_SCORE,
                          glove_lag=glove_lag, fitted=bool(measurements),
                          anchor=anchor)
    offline, fused_order, reset_before = offline_by_row(events, loaded)

    learned = learn_from_session(
        [g for entry in loaded for g in entry["glove"]],
        [c for entry in loaded for c in entry["cam"]],
        gates=gates, rail_params=rail_params, measurements=measurements)
    same_learning = (learned.scale == run.scale
                     and learned.curl_gates == run.curl_gates
                     and (rail_params is None or learned.rails == run.rails))

    print(f"Replay of {input_dir}: {len(loaded)} Ultraleap take(s)")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    for hand, row in sorted(lag_rows.items()):
        print(f"  {hand}: glove lag "
              + (f"{row.seconds:.3f} s ({row.source})" if row.applied
                 else "none applied"))
    for hand in sorted(measurements):
        print(f"  {hand}: template fitted")
    for hand, why in sorted(refusals.items()):
        print(f"  {hand}: template fit refused ({why})")
    print("  drift anchor: " + ("on" if anchor is not None else "off"))
    print("  rails, endpoints and curl gates handed to the live path: "
          + ("identical to fuse_all's" if same_learning
             else "DIFFERENT from fuse_all's"))

    # The takes in the order fuse_all fused them (the anchor's memory runs
    # from one take into the next), then any it never reached.
    drift = DriftAnchor(anchor) if anchor is not None else None
    order = fused_order + [k for k in range(len(loaded))
                           if k not in set(fused_order)]
    live = {}
    for k in order:
        entry = loaded[k]
        if drift is not None and reset_before.get(k):
            drift.reset()
        live[k] = replay_take(entry["glove"], entry["cam"], learned,
                              gates=gates, rail_params=rail_params,
                              unreliable=unreliable, anchor=drift,
                              lag=glove_lag or {}, max_dt=max_dt,
                              clock=entry["clock"], with_rows=True)

    print()
    print(f"  {'take':<44} {'frames':>7} {'paired off/live':>15} "
          f"{'max |diff| m':>13} {'mismatches':>10}")
    all_ok = same_learning
    for k, entry in enumerate(loaded):
        n, p_off, p_live, worst, bad = compare_take(entry, offline, live[k])
        ok = bad == 0 and p_off == p_live and worst <= REPLAY_TOL_M
        all_ok = all_ok and ok
        print(f"  {entry['name'][:44]:<44} {n:>7} {p_off:>7}/{p_live:<7} "
              f"{worst:>13.2e} {bad:>10}  {'OK' if ok else 'DIFFERS'}")
    print()
    print("Live path agrees with fuse_all on every take." if all_ok else
          "Live path DISAGREES with fuse_all: do not trust live output until "
          "this is explained.")
    return 0 if all_ok else 3


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    hands = ("left", "right") if args.hand == "both" else (args.hand,)
    osc_target = (parse_host_port(args.osc_out) if args.osc_out is not None
                  else None)

    # --- the profile, the masks and the lag --------------------------------
    gates = DEFAULT_GATES
    profile_path, profile_how = resolve_profile(args.profile)
    (unreliable, profile_rail, profile_name, _comment,
     profile_lag) = (load_profile(profile_path) if profile_path is not None
                     else ({}, {}, "", "", {}))
    cli_rail = (None if args.rail_fingers is None
                else parse_rail_fingers(args.rail_fingers))
    if cli_rail is not None:
        rail_spec = cli_rail
    elif profile_rail:
        rail_spec = profile_rail
    else:
        rail_spec = DEFAULT_RAIL.fingers
    rail_params = (None if args.no_rail_override
                   else RailOverrideParams(fingers=rail_spec))
    lags = resolve_lag(parse_glove_lag(args.glove_lag), hands, profile_lag,
                       profile_path)
    anchor = (DriftAnchorParams(window_s=args.anchor_window,
                                hold_s=args.anchor_hold,
                                deadband=args.anchor_deadband)
              if args.drift_anchor == "on" else None)
    fit_spec = str(args.fit_template).strip()
    if args.replay is not None:
        return replay_session(args.replay, gates, rail_params, unreliable,
                              profile_lag, anchor, fit_spec, args.glove_lag,
                              args.max_dt)
    if fit_spec.lower() in (FIT_AUTO, FIT_NONE):
        fit = fit_spec.lower()
    else:
        try:
            fit = load_measurement(fit_spec)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise SystemExit(f"--fit-template {fit_spec}: {e}")

    log = RunLog(args.out, header=run_header(
        "fuse_live.py", hands, profile_path, profile_how, lags, fit_spec))
    log.say(f"Profile: {profile_path or '(none)'}  [{profile_how}]"
            + (f"  {profile_name}" if profile_name else ""))
    for hand in hands:
        seconds, why = lags[hand]
        masked = ", ".join(unreliable.get(hand, ())) or "(none)"
        override = ("off (--no-rail-override)" if rail_params is None
                    else ", ".join(rail_params.fingers_for(hand)) or "(none)")
        log.say(f"  {hand}: glove lag {seconds:.3f} s ({why}); unreliable "
                f"{masked}; rail override {override}")
    log.say(f"Template fit: {fit_spec}.  Drift anchor: {args.drift_anchor}.  "
            f"Pairing within {args.max_dt:g} s.")

    # --- the sensors ---------------------------------------------------------
    buffer = CameraBuffer(gates=gates, keep_abs26=True)
    camera = CameraSource(hands, mock=args.mock_leap)
    glove = GloveSource(hands, mock=args.mock_glove, port=args.port)
    try:
        camera.start()
    except Exception as e:                  # LeapUnavailable says what to do
        log.say(f"camera: {e}")
        log.write_warmup()
        return 1
    try:
        glove.start()
    except OSError as e:
        camera.stop()
        log.say(f"glove: cannot listen on OSC port {args.port} ({e}). Is "
                "another recorder still running?")
        log.write_warmup()
        return 1
    view = CameraView(hand=args.hand if args.hand != "both" else None,
                      band=DEFAULT_BAND,
                      enabled=not (args.no_view or args.mock_leap)).start()
    beeper = AsyncBeeper(beep)
    hud = Hud(lambda s: print(s, end="", flush=True), every=HUD_EVERY)
    sinks = []

    def shutdown():
        hud.close()
        for closer in (glove.stop, camera.stop, view.close, beeper.stop,
                       *[s.close for s in sinks]):
            try:
                closer()
            except Exception:
                pass

    # --- the warm-up ------------------------------------------------------
    warmup = Warmup(hands, open_s=args.warmup_open, fist_s=args.warmup_fist,
                    acquire_s=args.acquire_timeout, countdown_s=COUNTDOWN_S,
                    band=DEFAULT_BAND)
    words = band_words(DEFAULT_BAND)
    log.say("Warm-up, one hand at a time ("
            + ", then ".join(h.upper() for h in hands)
            + f"): acquire, open palm {args.warmup_open:g} s, fist "
            f"{args.warmup_fist:g} s.")
    shown = [None]
    next_hud = [0.0]
    asked = {}

    def caption(text):
        # The window's caption is a status file; write it only on a change.
        if text != shown[0]:
            shown[0] = text
            view.caption(text, band=DEFAULT_BAND)

    def cue(stage, hand, seconds):
        name = hand.upper()
        camera.coach(MOCK_POSE.get(stage))
        glove.coach(hand, stage)
        hud.close()
        if stage == STAGE_ACQUIRE:
            asked[hand] = time.time()
            caption(f"{name} hand: ACQUIRE, open palm {words} up")
            log.say(f"{name} hand: ACQUIRE. Hold the {name} hand open over "
                    f"the module, palm to the lens, {words} up (waiting up "
                    f"to {seconds:g} s).")
        elif stage == STAGE_COUNTDOWN:
            beeper.beep(ACQUIRED_FREQ, CUE_MS)
            now = time.time()
            log.say(acquired_line(hand, now - asked.get(hand, now)))
            caption(countdown_text(hand, seconds))
            hud.show(countdown_text(hand, seconds), now, force=True)
            next_hud[0] = now + HUD_EVERY
        elif stage == STAGE_NOT_ACQUIRED:
            caption(f"{name} hand: NOT ACQUIRED")
            later = hands[hands.index(hand) + 1:]
            log.say(f"{name} hand refused: {warmup.not_acquired[hand]}"
                    + (f". Going on to the {later[0].upper()} hand."
                       if later else ""))
        else:
            beeper.beep(CUE_FREQ, CUE_MS)
            caption(f"{name} hand: {stage} ({seconds:g} s)")
            log.say(f"{name} hand: {stage} for {seconds:g} s")

    def poll(w):
        for row in camera.drain():
            w.add_camera(buffer.add(row))
        for g in glove.drain():
            w.add_glove(g["hand_side"], flexion_features(g["pts"]),
                        g["frame"])

    def acquire(hand, now):
        other = "right" if hand == "left" else "left"
        return acquire_status(hand, glove.recent_packets(hand, now),
                              glove.recent_packets(other, now),
                              camera.reading(hand, now),
                              camera.seen_recently(other, now),
                              band=DEFAULT_BAND)

    def show(stage, hand, left, status):
        now = time.time()
        if now < next_hud[0]:
            return
        next_hud[0] = now + HUD_EVERY
        if stage == STAGE_ACQUIRE:
            line = status.line(left)
            caption(f"{hand.upper()} hand: ACQUIRE, {status.caption_text}")
        elif stage == STAGE_COUNTDOWN:
            line = countdown_text(hand, left)
            caption(line)
        else:
            hz = glove.rate_hz(hand)
            fresh = camera_fresh(buffer, hand, now)
            line = (f"{hand.upper()} {stage[:9]:<9} {max(0.0, left):4.1f}s  "
                    f"glove {'--' if hz is None else f'{hz:.0f}'}/s  "
                    f"camera {'fresh' if fresh else 'STALE'}")
        hud.show(line, now, force=True)

    try:
        warmup.run(poll, cue, show, acquire=acquire)
    except KeyboardInterrupt:
        shutdown()
        log.say("\nStopped during the warm-up; nothing was fused.")
        log.write_warmup()
        return 1
    except BaseException:
        shutdown()
        log.write_warmup()
        raise
    hud.close()
    camera.coach(None)
    buffer.keep_abs26 = False
    learned = warmup.learn(gates=gates, rail_params=rail_params, fit=fit)
    del warmup
    learned, fused_hands = conclude_warmup(learned, hands, fit, args.out, log)
    if not fused_hands:
        shutdown()
        return 2

    # --- the fusion ---------------------------------------------------------
    fusion = LiveFusion(learned, buffer, gates=gates, rail_params=rail_params,
                        unreliable=unreliable, anchor=anchor,
                        lag={h: lags[h][0] for h in fused_hands},
                        max_dt=args.max_dt)
    if args.out is not None:
        sinks.append(JsonlSink(args.out))
    if osc_target is not None:
        sinks.append(OscSink(*osc_target))
    beeper.beep(CUE_FREQ, CUE_MS)
    view.caption("LIVE FUSION", band=DEFAULT_BAND)
    print("Fusing" + (f" for {args.seconds:g} s" if args.seconds else
                      " until Ctrl-C") + ".")
    log.fusing()
    latest = {}
    t_end = None if args.seconds is None else time.time() + args.seconds
    try:
        while t_end is None or time.time() < t_end:
            # Camera first, so the buffer holds the newest frames before
            # the glove frames that may want them are paired.
            for row in camera.drain():
                buffer.add(row)
            for g in glove.drain():
                out = fusion.step(g)
                if out is None:
                    continue
                latest[out.hand] = out
                for sink in sinks:
                    sink.write(out)
            now = time.time()
            if now >= next_hud[0]:
                # Built only when it will be shown: the curls are computed
                # four times a second, not on every pass of the loop.
                next_hud[0] = now + HUD_EVERY
                hud.show(hud_line([hud_segment(h, latest.get(h),
                                               glove.rate_hz(h),
                                               camera_fresh(buffer, h, now))
                                   for h in fused_hands]), now, force=True)
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    report_run(log, fusion, sinks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
