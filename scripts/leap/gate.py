"""The Phase 2 gate: can the IR camera see a hand inside the glove?

This is the experiment `docs/ultraleap_ir170_plan.md` section 2 describes,
run end to end by one command. Per condition it announces the condition,
counts you in with beeps, runs a TIMED POSE SCHEDULE while recording, takes IR
stills spread across the run, and measures the numbers the plan's decision
table turns on. Between conditions it prints what to change on your hand and
waits for Enter, because by then you are wearing a glove and cannot read a
script.

The schedule is what makes the comparison mean anything. On 2026-09-17 the
operator chose poses and heights freely, so the bare baseline scored 62 %
because the hand turned edge-on for five seconds, one gloved run sat 99 mm
above the lens, and four gloved runs lost tracking while the hand was a fist —
with no bare-hand fist anywhere to compare against. Now every condition holds
`--schedule` (default `open_palm:5,fist:5,pinch:5,spread:5`), each frame
carries the plan and its own offset into it, and the report compares fist
against fist. A live one-line HUD shows the pose being called for, whether a
hand is tracked, the palm height against `--band` and the viewing angle; out
of band is flagged and counted, never blocked.

    bare           nothing on the hand — the reference every other row is
                   measured against
    glove          the black StretchSense glove
    glove_liner    a thin white cotton liner glove pulled over it
    glove_tape     white tape strips on fingertips and knuckles

What it writes:

    recordings/leap/gate/<condition>/<condition>_<hands>_take1_<stamp>.jsonl
    recordings/leap/gate/<condition>/<condition>_<k>_L.png   (+ _R.png, .json)
    recordings/leap/gate/<condition>/<same name>.lmt         (with --raw)
    results/leap_gate/REPORT.txt

`REPORT.txt` is the deliverable: one table row per condition and hand, then
the verdict as a sentence — "Path A: yes because ..." or "Path A: no because
...". The IR stills are the evidence behind it, each with a sidecar saying
whether the tracker reported a hand at that instant.

  python scripts/leap/gate.py                            # the full protocol
  python scripts/leap/gate.py --conditions bare,glove --seconds 30
  python scripts/leap/gate.py --raw                      # also write .lmt
  python scripts/leap/gate.py --mock --seconds 3         # dry run, no camera
  python scripts/leap/gate.py --recompute                # report from disk

`--recompute` is the one that touches no hardware: it rebuilds the report
from the recordings and IR stills already under `--out-dir`, discovering the
conditions from the folder names. Use it when a measurement changes — the
detection denominator did, see `leap_hand.stats.choose_rate` — so an earlier
session gets the corrected numbers without anyone re-running the protocol.
Where a condition folder holds several takes it measures the most recent and
names the others in the report. Takes recorded before schedules existed have
no pose boundaries in them: `--recompute` reads them exactly as it always did,
prints the per-condition table, and says why there is no per-pose block rather
than attributing frames to a pose by where they sit in the file.

Condition names are free-form. `bare` is the reference; everything else is a
gloved condition judged against the bare hand of the same side, which is how
the reviewer's extra runs fit with no code change:

  python scripts/leap/gate.py --conditions bare,glove_right,glove_both

Protocol (plan section 6): module flat on the table, lenses up, hand 20 to 50
cm above it, palm roughly facing the camera, no sunlight and no other IR
sources. Control Panel: Desktop mode, Allow images on, Balanced performance,
interpolation off. Ctrl+C keeps everything recorded so far and still writes
the report.
"""
import argparse
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from leap_hand.gate import (
    DEFAULT_CONDITIONS,
    ConditionResult,
    attach_pose_stats,
    format_report,
    instruction_for,
    scan_out_dir,
    verdict,
)
from leap_hand.images import HandTrail, open_sampler, write_snapshot
from leap_hand.protocol import (
    DEFAULT_BAND,
    DEFAULT_POSES,
    AsyncBeeper,
    Hud,
    band_for_condition,
    band_text,
    default_schedule,
    hud_line,
    parse_band,
    parse_schedule,
    read_hand,
)
from leap_hand.recorder import LeapRecorder
from leap_hand.stats import analyse_paths
from leap_hand.stream import LeapUnavailable, open_stream
from xr_hand.recorder import finalize_pose_name, hand_tag, pose_filename, slugify

DEFAULT_OUT = Path("recordings") / "leap" / "gate"
DEFAULT_REPORT = Path("results") / "leap_gate" / "REPORT.txt"
MOCK_NOTE = "synthetic: --mock run, not camera data"
MIN_VISIBLE_TIME_US = 300_000     # plan section 6: a hand counts after 0.3 s
MAX_DRAIN_ROUNDS = 8              # bound on the post-beep flush
HUD_STALE_S = 0.30                # a reading older than this is "no hand"
# The poses MockLeapStream can actually act out; anything else leaves it
# cycling, which is honest — the mock is not pretending to pinch.
MOCK_POSES = ("open_palm", "fist", "spread", "thumb_opposition")


def beep(freq: int = 880, ms: int = 180) -> None:
    """Audible cue; falls back to the terminal bell off Windows."""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


class GateRun:
    """One condition of the gate: record, snapshot, measure."""

    def __init__(self, source, sampler, out_dir: Path, hz, raw: bool,
                 mock: bool, schedule=None, band=None, hand=None):
        self.source = source
        self.sampler = sampler
        self.out_dir = out_dir
        self.hz = hz
        self.raw = raw
        self.mock = mock
        self.schedule = schedule or default_schedule()
        self.band = band                 # None = let the condition name decide
        self.hand = hand                 # None = whichever hand turns up
        self.trail = HandTrail()
        self.skipped_young = 0
        self.beeper = AsyncBeeper(beep)
        self.hud = Hud(self._write)
        self._reading = None
        self._reading_at = 0.0
        self._other_at = 0.0

    # --- plumbing --------------------------------------------------------
    @staticmethod
    def _write(text: str) -> None:
        import sys
        sys.stdout.write(text)
        sys.stdout.flush()

    def _consume(self, recorder=None) -> int:
        """Drain pending hands into the trail, and into the recorder if any."""
        seen = 0
        now = time.time()
        for _side, lh in self.source.drain(64):
            seen += 1
            self.trail.add(lh)
            if self.hand and lh.hand_side != self.hand:
                self._other_at = now
            else:
                self._reading, self._reading_at = read_hand(lh), now
            # Gate on presence and settling time, never on confidence: LeapC
            # documents confidence as a constant 1.0 (plan section 6).
            if lh.visible_time_us < MIN_VISIBLE_TIME_US:
                self.skipped_young += 1
                continue
            if recorder is not None:
                recorder.record(lh)
        return seen

    def _show_hud(self, phase: str, seconds_left, band, pose: str = "") -> None:
        now = time.time()
        fresh = self._reading if now - self._reading_at < HUD_STALE_S else None
        expected = self.hand or (fresh.hand_side if fresh else "hand")
        self.hud.show(
            hud_line(phase, seconds_left, fresh, expected, band,
                     saw_other_hand=now - self._other_at < HUD_STALE_S,
                     extra=f"HOLD: {pose.replace('_', ' ').upper()}" if pose
                     else ""),
            now)

    def _say(self, *lines: str) -> None:
        self.hud.close()
        for line in lines:
            print(line, flush=True)

    def _discard_backlog(self) -> int:
        """Drop what queued while the start beep blocked.

        `winsound.Beep` blocks for its whole duration while LeapC's polling
        thread keeps filling the queue, so hands drained straight after a
        250 ms beep are up to 250 ms old and would be written with the
        current time. Here that matters twice over: those frames inflate the
        count at the head of the take without lengthening its span, and the
        count over the span IS the detection rate this experiment turns on.
        The trail still sees every hand, so the IR stills keep pairing.
        """
        dropped = 0
        for _ in range(MAX_DRAIN_ROUNDS):
            n = self._consume()
            dropped += n
            if not n:
                break
        return dropped

    def _raw_capture(self, path: Path):
        if not self.raw:
            return nullcontext()
        from leap_hand.replay import RawRecording
        return RawRecording(self.source, path)

    def _snapshot(self, folder: Path, condition: str, index: int):
        pair = self.sampler.wait_for_pair(timeout=1.5)
        if pair is None:
            self._say(f"      [still {index}] no image within 1.5 s — is "
                      "'Allow images' on in the Control Panel?")
            return None
        self._consume()
        hands, dt_ms = self.trail.nearest(pair.timestamp_us)
        snap = write_snapshot(folder, condition, index, pair, hands=hands,
                              tracking_dt_ms=dt_ms,
                              note=MOCK_NOTE if self.mock else "")
        self._say(f"      [still {index}] {pair.size_text}  "
                  f"{snap.tracking_text}")
        return snap

    # --- one condition ---------------------------------------------------
    def run(self, condition: str, prep: float, snapshots: int
            ) -> ConditionResult:
        folder = self.out_dir / condition
        band = band_for_condition(condition, self.band)
        seconds = self.schedule.total
        # Nothing the previous condition saw may be paired with this one's
        # photographs: a still of an untracked glove must never inherit the
        # bare hand from ten minutes ago.
        self.trail.clear()
        self._say(f"--- Condition: {condition.upper()}  ({seconds:g} s, "
                  f"{snapshots} IR still(s)) ---",
                  f"    {instruction_for(condition)}",
                  f"    Height band {band_text(band)}, palm toward the "
                  "camera. Hold each pose as it is called:",
                  f"      {self.schedule.text}")

        for s in range(int(round(prep)), 0, -1):
            if s <= 3:
                self._say(f"      {s}...")
                self.beeper.beep(660, 120)
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self._consume()
                self._show_hud("PREP", t_end - time.time(), band,
                               self.schedule.windows[0].pose)
                time.sleep(0.02)

        recorder = LeapRecorder(hz=self.hz, pose=condition, take=1,
                                pose_plan=self.schedule.text)
        path = folder / pose_filename(condition, 1)
        raw_path = path.with_suffix(".lmt")
        # Stills spread across the run, the first one a beat after the start
        # so the tracker has settled and the last well before the end.
        due = [seconds * (k + 0.5) / snapshots for k in range(snapshots)]
        snaps = []

        with self._raw_capture(raw_path):
            # The start beep is on the beeper thread now, so there is no
            # quarter second of backlog behind it — but the flush stays,
            # because anything the countdown left queued is still older than
            # the take (see `_discard_backlog`).
            self.beeper.beep(1000, 250)
            self._discard_backlog()
            recorder.start(path)
            t0 = time.time()
            # The schedule's clock and the recorder's are the same instant, so
            # `pose_t` on a frame is exactly the offset the HUD is showing.
            recorder.schedule_t0 = t0
            try:
                t_end = t0 + seconds
                taken = 0
                held = None
                while time.time() < t_end:
                    self._consume(recorder)
                    now = time.time()
                    window = self.schedule.at(now - t0)
                    pose = window.pose if window else ""
                    if window is not None and window is not held:
                        held = window
                        # The mock can act the pose out, so a dry run's
                        # per-pose table is about the poses it names.
                        if hasattr(self.source, "set_pose"):
                            self.source.set_pose(pose if pose in MOCK_POSES
                                                 else None)
                        self.beeper.beep(1200 if window.index else 1000, 150)
                        self._say(f"      NOW: {pose.replace('_', ' ').upper()}"
                                  f"   ({window.seconds:g} s)")
                    if taken < len(due) and now - t0 >= due[taken]:
                        snap = self._snapshot(folder, condition, taken)
                        if snap is not None:
                            snaps.append(snap)
                        taken += 1
                    self._show_hud("REC", t_end - time.time(), band, pose)
                    time.sleep(0.005)
            finally:
                # Runs on Ctrl+C too: close the file, then keep or drop it.
                self.hud.close()
                recorder.stop()
                self.beeper.beep(500, 300)

        result = ConditionResult(condition=condition, snapshots=len(snaps),
                                 snapshots_with_hand=sum(1 for s in snaps
                                                         if s.saw_hand),
                                 folder=str(folder))
        if recorder.count == 0:
            # Drop the empty JSONL, but KEEP the .lmt: an .lmt full of
            # hand-less tracking events is the proof that the camera streamed
            # for the whole run and reported nothing, which in this experiment
            # is a result rather than a missing file.
            path.unlink(missing_ok=True)
            kept_raw = self.raw and raw_path.exists()
            result.note = ("no frames recorded: the tracker reported no hand "
                           "for the whole run"
                           + (f"; raw stream kept in {raw_path.name}"
                              if kept_raw else ""))
            print(f"      no hand tracked in {seconds:g} s "
                  f"({len(snaps)} IR still(s) kept"
                  f"{', .lmt kept' if kept_raw else ''})\n")
            return result

        final = finalize_pose_name(path, recorder.hands_seen)
        if self.raw and raw_path.exists():
            raw_path.replace(final.with_suffix(".lmt"))
        result.recordings = [str(final)]
        result.stats = analyse_paths([final])
        attach_pose_stats(result, final, self.band)
        self._say(f"      saved {recorder.count} frames "
                  f"({hand_tag(recorder.hands_seen)}) -> {final.name}, "
                  f"{len(snaps)} IR still(s)")
        for s in result.pose_stats:
            self._say(f"        {s.hand_side:<5} {s.pose:<10} "
                      f"{s.detection_rate * 100:5.1f}% detected, longest loss "
                      f"{s.longest_loss_s:.2f} s")
        print()
        return result


def wait_for_operator(condition: str, mock: bool) -> None:
    """Print what to change on the hand, then wait — unless this is a dry run."""
    print("=" * 62)
    print(f"NEXT: {condition}")
    print(f"  {instruction_for(condition)}")
    if mock:
        print("  (--mock: not waiting)")
        print("=" * 62 + "\n")
        return
    print("=" * 62)
    try:
        input("  Press Enter when the hand is ready... ")
    except EOFError:
        print("  (no console to read from — continuing)")
    print()


def write_report(results, report_path: Path, title: str,
                 seconds=None, band=None) -> str:
    """Judge, format, write and print. The one place a report is produced."""
    v = verdict(results)
    report = format_report(results, v, title=title, seconds=seconds, band=band)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print(f"\nwrote {report_path}")
    return report


def recompute(out_dir: Path, report_path: Path, conditions=None,
              band=None) -> None:
    """Rebuild the report from the files already in `out_dir`. No camera.

    The conditions are whatever sub-folders are there, so a run with the
    reviewer's names (`glove_right`, `glove_20cm`, ...) needs nothing said
    here; `--conditions` only narrows the set.
    """
    print("=" * 62)
    print("Ultraleap Phase 2 gate — recompute (no camera, no recording)")
    print(f"  reading: {out_dir}    report: {report_path}")
    print("=" * 62)

    if not out_dir.is_dir():
        raise SystemExit(f"no such folder: {out_dir}")
    results = scan_out_dir(out_dir, conditions, band)
    if not results:
        raise SystemExit(
            f"no condition folders under {out_dir}. A gate run writes one "
            "folder per condition; run the gate first, or point --out-dir at "
            "the folder that holds them.")
    if conditions:
        missing = [c for c in conditions
                   if c not in {r.condition for r in results}]
        if missing:
            print(f"  (no folder for: {', '.join(missing)})")

    for r in results:
        takes = len(r.recordings)
        plan = (f", schedule {r.schedule_text}" if r.scheduled
                else ", no pose schedule")
        print(f"  {r.condition:<16} {r.frames:>6} frames from "
              f"{takes} take(s), {r.snapshots} IR still(s){plan}")
    if results and not any(r.scheduled for r in results):
        print("  (none of these takes carries a pose schedule, so the report "
              "falls back to the per-condition table)")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_report(results, report_path,
                 title=f"recomputed: {stamp}   from: {out_dir}   "
                       "(files on disk, no camera)", band=band)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the Phase 2 gate: does the IR camera see a hand "
                    "inside the StretchSense glove?")
    p.add_argument("--conditions", default=None,
                   help="comma-separated conditions, any names you like "
                        f"(default: {','.join(DEFAULT_CONDITIONS)}). With "
                        "--recompute it narrows what is read from --out-dir; "
                        "left out there, every folder is read.")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="seconds recorded per condition (default: 20). "
                        "Ignored when --schedule is given: the schedule's own "
                        "total is the length of the run.")
    p.add_argument("--schedule", default=None,
                   help="the timed pose schedule every condition holds, as "
                        "pose:seconds pairs (default: "
                        + ",".join(f"{p}:5" for p in DEFAULT_POSES)
                        + "). This is what makes glove and bare comparable "
                          "pose for pose.")
    p.add_argument("--band", default=None,
                   help="palm height band LOW,HIGH in centimetres (default: "
                        f"{DEFAULT_BAND[0]:g},{DEFAULT_BAND[1]:g}, or N+/-5 "
                        "for a condition named <name>_<N>cm). Out of band is "
                        "flagged and counted, never blocked.")
    p.add_argument("--hand", default=None, choices=("left", "right"),
                   help="the hand this run is about, for the HUD's WRONG HAND "
                        "warning. The report always pairs by the hand in the "
                        "frames, whether or not this is given.")
    p.add_argument("--prep", type=float, default=5.0,
                   help="countdown seconds before each run (default: 5)")
    p.add_argument("--snapshots", type=int, default=3,
                   help="IR stills per condition, spread across the run "
                        "(default: 3)")
    p.add_argument("--hz", type=float, default=0.0,
                   help="frames saved per second (default: 0 = keep every "
                        "frame, which is what the gate metrics want)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"recordings root (default: {DEFAULT_OUT})")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                   help=f"report path (default: {DEFAULT_REPORT})")
    p.add_argument("--raw", action="store_true",
                   help="also write LeapC's own .lmt beside each take")
    p.add_argument("--mock", action="store_true",
                   help="dry run with synthetic hands and gradient images")
    p.add_argument("--recompute", action="store_true",
                   help="rebuild the report from what is already in "
                        "--out-dir; no camera, no recording")
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"))
    p.add_argument("--timeout", type=float, default=5.0,
                   help="seconds to wait for a device (default: 5)")
    args = p.parse_args()

    given = args.conditions if args.conditions is not None else ",".join(
        DEFAULT_CONDITIONS)
    conditions = [slugify(c) for c in given.split(",") if slugify(c)]
    if not conditions:
        raise SystemExit("no conditions given")
    if args.snapshots < 0:
        raise SystemExit("--snapshots cannot be negative")
    if args.raw and args.mock:
        raise SystemExit("--raw needs a live camera: there is no LeapC stream "
                         "behind --mock")
    try:
        band = parse_band(args.band) if args.band else None
        schedule = (parse_schedule(args.schedule) if args.schedule
                    else default_schedule(args.seconds))
    except ValueError as e:
        raise SystemExit(str(e))

    if args.recompute:
        if args.raw or args.mock:
            raise SystemExit("--recompute reads files already on disk: "
                             "--raw and --mock have nothing to do there")
        wanted = conditions if args.conditions is not None else None
        recompute(args.out_dir, args.report, wanted, band)
        return

    seconds = schedule.total
    eta = len(conditions) * (args.prep + seconds)
    print("=" * 62)
    print("Ultraleap Phase 2 gate" + ("  [mock]" if args.mock else ""))
    print(f"  conditions: {', '.join(conditions)}")
    print(f"  schedule:   {schedule.text}   ({seconds:g} s per condition)")
    print(f"  band:       {band_text(band) if band else band_text(DEFAULT_BAND)}"
          + ("" if band else ", or N+/-5 cm for a <name>_<N>cm condition"))
    print(f"  {args.snapshots} IR still(s) each "
          f"(~{eta / 60:.1f} min plus changeovers)")
    print(f"  recordings: {args.out_dir}    report: {args.report}")
    print("=" * 62 + "\n")

    try:
        source = open_stream(mock=args.mock, mode=args.mode,
                             device_timeout=args.timeout)
    except LeapUnavailable as e:
        raise SystemExit(f"\nNo live tracking: {e}\n")

    try:
        sampler = open_sampler(source, mock=args.mock)
    except LeapUnavailable as e:
        source.stop()
        raise SystemExit(f"\nNo IR images: {e}\n")

    run = GateRun(source, sampler, args.out_dir, args.hz or None, args.raw,
                  args.mock, schedule=schedule, band=band, hand=args.hand)
    results = []
    try:
        for condition in conditions:
            wait_for_operator(condition, args.mock)
            results.append(run.run(condition, args.prep, args.snapshots))
    except KeyboardInterrupt:
        print("\nInterrupted — reporting on the conditions finished so far.")
    finally:
        run.hud.close()
        run.beeper.stop()
        sampler.detach(getattr(source, "connection", None))
        source.stop()

    if not results:
        raise SystemExit("Nothing recorded; no report written.")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = (f"run: {stamp}{'  [MOCK - synthetic data, not evidence]' if args.mock else ''}"
             f"   device: {getattr(source, 'device_serial', None) or 'unknown'}"
             f"   schedule: {schedule.text}")
    write_report(results, args.report, title=title, seconds=seconds, band=band)
    if run.skipped_young:
        print(f"  ({run.skipped_young} hands skipped: tracked for less than "
              f"{MIN_VISIBLE_TIME_US / 1000:.0f} ms — settling)")


if __name__ == "__main__":
    main()
