"""The 60 second tracking test: why does the camera lose the hand?

Hardware setup: camera only, bare hand. The Stereo IR 170 flat on the desk,
lenses up, desktop mode; no glove.

Why this exists. The operator's words were "the hand tracking for the camera
is not good and it does not register sometimes". The tracking service is
healthy (its log shows 90 Hz, no dropped frames, about 10 ms of latency, the
config unchanged since install) and earlier bare-hand static poses tracked
97 to 100 percent of their frames. So the losses happen in the moment: the
hand too high, too low or out at the edge of the field, the palm turned
away, a closed hand seen from below, a hand moving fast, sunlight or another
infrared source, smudged lenses, a device warning. Once the moment has
passed nothing says which. This script coaches the operator through one
minute that visits every one of those on purpose, slowly, records every
tracking frame (empty ones included) with the IR image brightness and the
device's own status flags beside it, and then explains each loss:

  results\\diagnostics\\tracking_quality_<stamp>.csv   one row per tracking frame
  results\\diagnostics\\tracking_quality_<stamp>.txt   the report printed at the end

The report gives the tracked percentage, the frame rate, every loss (gone
longer than 100 ms, or a new hand id) with where the hand was and why, the
causes ranked by how many losses each explains, and the fix to try first.
The last line printed starts with `VERDICT:`. The classifier is
`leap_hand.tracking_quality`, the same one the camera window's "tracking:"
line and the grasp recorder's reject reasons use.

The live camera window opens beside it (a second, read-only client of the
tracking service) with the step and the running numbers as its caption.

Usage:
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --hand left
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --seconds 60 --hand left --out results\\diagnostics
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --mock --seconds 5 --no-view
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --from results\\diagnostics\\tracking_quality_<stamp>.csv

`--mock` needs no hardware: a scripted mock hand is lost on purpose in every
way the report can explain, so the whole report is exercised. `--from`
re-reads a saved CSV (or a recorded take's JSONL) and prints the report
again, for after the fact.
"""
import argparse
import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import List, Optional

from leap_hand.protocol import AsyncBeeper, CameraView, Hud
from leap_hand.stream import LeapStream, LeapUnavailable
from leap_hand.tracking_quality import (
    LOSS_GAP_S,
    TARGET_CM,
    LiveLossTracker,
    MockScenario,
    SERIES_COLUMNS,
    Sample,
    cadence_gap_s,
    cause_label,
    decode_device_status,
    format_report,
    image_stats,
    rows_to_samples,
    samples_from_series_rows,
    series_rows,
    state_from_leaphand,
    summarise,
)

DEFAULT_OUT = Path("results") / "diagnostics"
GET_READY_S = 3.0
IMAGE_EVERY_S = 0.1          # IR brightness is measured ten times a second
IMAGE_FRESH_S = 0.25         # older than this is not attached to a frame
HUD_EVERY_S = 0.25

# The coached minute, as seconds of a 60 s run; a shorter or longer run is
# scaled. Every step is something that has cost tracking before, done slowly
# so the moment it fails is visible and recorded.
STEPS = (
    (0.0, "HOLD AT 30 CM", "hold the hand open, palm toward the lenses, 30 cm above "
                           "the middle of the module, and keep it still"),
    (10.0, "UP TO 50 CM, SLOWLY", "move the hand slowly up to 50 cm, then slowly back "
                                  "down to 30 cm"),
    (20.0, "OUT TO EACH SIDE", "at 30 cm, move slowly out to the left and back, then "
                               "out to the right and back"),
    (30.0, "TURN THE PALM AWAY", "turn the palm slowly away from the lenses, then back"),
    (38.0, "FIST, SLOWLY", "close the hand slowly into a fist, then open it"),
    (43.0, "PINCH, SLOWLY", "pinch the thumb and index slowly, then open"),
    (48.0, "YOUR GRASPS, SLOWLY", "hold each of the paper grasps you like, one after "
                                  "another, slowly"),
)


@dataclass
class FrameRecord:
    """One tracking frame as it came off the camera, before it is reduced."""

    t: float                                 # seconds, the camera's own clock
    frame_id: Optional[int]
    framerate: Optional[float]
    hands: list = field(default_factory=list)          # LeapHand
    image_mean: Optional[float] = None
    image_saturated: Optional[float] = None
    status: Optional[tuple] = None


def to_sample(rec: FrameRecord) -> Sample:
    return Sample(t=rec.t, hands=tuple(state_from_leaphand(lh) for lh in rec.hands),
                  framerate=rec.framerate, image_mean=rec.image_mean,
                  image_saturated=rec.image_saturated, status=rec.status,
                  frame_id=rec.frame_id)


# --- the real camera ------------------------------------------------------------
class QualityStream(LeapStream):
    """`LeapStream` plus what a loss report needs and a plain stream drops.

    Every tracking frame, the empty ones included (LeapStream only queues
    hands, and a loss is exactly a run of frames with no hand); the device's
    own status flags, from the device event at connect and every status
    change after it; dropped-frame events; and the IR image brightness. All
    of it arrives on the SDK's thread, so it is only stored here, never
    printed or drawn.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._records: deque = deque(maxlen=90 * 600)
        self.status: Optional[tuple] = None      # None until a device event says
        self.status_raw: Optional[int] = None
        self.status_events = 0
        self.status_history: List[tuple] = []
        self.dropped_frame_events = 0
        self.images = 0
        self.image_errors = 0
        self._image = None                       # (mean, saturated, wall time)

    def _note_status(self, event) -> None:
        try:
            raw = int(event.c_data.status)
        except Exception:                        # pragma: no cover - hardware path
            return
        self.status_raw = raw & 0xFFFFFFFF
        self.status = decode_device_status(raw)
        self.status_events += 1
        self.status_history.append((time.time(), self.status_raw, self.status))

    def _on_image(self, event) -> None:
        now = time.time()
        if self._image is not None and now - self._image[2] < IMAGE_EVERY_S:
            return
        try:
            from leap_hand.images import image_to_numpy
            mean, sat = image_stats(image_to_numpy(event.image[0]))
        except Exception:                        # pragma: no cover - hardware path
            self.image_errors += 1
            return
        self.images += 1
        self._image = (mean, sat, now)

    def _build_listener(self):
        base = super()._build_listener()
        stream = self

        class _QualityListener(type(base)):
            def on_device_event(self, event):
                stream._note_status(event)
                super().on_device_event(event)

            def on_device_status_change_event(self, event):
                stream._note_status(event)

            def on_device_failure_event(self, event):
                stream._note_status(event)

            def on_dropped_frame_event(self, event):
                stream.dropped_frame_events += 1

            def on_image_event(self, event):
                stream._on_image(event)

        return _QualityListener()

    def _on_tracking(self, event) -> None:
        super()._on_tracking(event)            # converts and queues the hands
        hands = []
        while True:
            try:
                hands.append(self.queue.get_nowait()[1])
            except Empty:
                break
        fid = getattr(event, "tracking_frame_id", None)
        hands = [lh for lh in hands if fid is None or lh.frame_id == fid]
        img = self._image
        if img is not None and time.time() - img[2] > IMAGE_FRESH_S:
            img = None
        rate = float(getattr(event, "framerate", 0.0) or 0.0)
        self._records.append(FrameRecord(
            t=float(event.timestamp) / 1e6, frame_id=fid, framerate=rate or None,
            hands=hands, image_mean=None if img is None else img[0],
            image_saturated=None if img is None else img[1], status=self.status))

    def _discard_pending(self) -> None:
        super()._discard_pending()
        self._records.clear()

    def records(self) -> List[FrameRecord]:
        out = []
        while True:
            try:
                out.append(self._records.popleft())
            except IndexError:
                return out

    def enable_images(self) -> str:
        """Ask for the IR images; returns a note when the answer is not yes.

        The answer is not trusted either way: whether images are there is
        decided at the end by whether any arrived (see
        `leap_hand.protocol.image_banner` for why the reply alone misled).
        """
        try:
            flags = self.connection.set_policy_flags(
                flags_to_set=[self._leap.enums.PolicyFlag.Images])
        except Exception as e:                   # pragma: no cover - hardware path
            return f"the image policy request failed ({type(e).__name__}: {e})"
        names = {getattr(f, "name", str(f)) for f in (flags or [])}
        return "" if "Images" in names else (
            f"the service did not confirm the image policy (flags: "
            f"{', '.join(sorted(names)) or 'none'})")


# --- the mock camera ------------------------------------------------------------
class MockSource:
    """`MockLeapStream` acting out `MockScenario`, one record per frame.

    The mock emits nothing for a dropped frame, the way LeapC emits no hand,
    so the frames in between are filled in as empty ones: every frame index
    is accounted for, as the real stream's empty tracking events are.
    """

    def __init__(self, seconds: float, hand: Optional[str]):
        from leap_hand.mock import MockLeapStream
        self.scenario = MockScenario(seconds)
        self.hz = self.scenario.hz
        self.mock = MockLeapStream(hz=self.hz, dropout_every=0, reacquire_every=0,
                                   sides=(hand or "left",), script=self.scenario.script)
        self.device_serial = self.mock.device_serial
        self._clock: Optional[float] = None

    def start(self) -> None:
        self._clock = time.time()

    def stop(self) -> None:
        self._clock = None

    def records(self) -> List[FrameRecord]:
        """The frames due by now, paced against the wall clock."""
        now = time.time()
        n = int((now - self._clock) * self.hz)
        if n <= 0:
            return []
        self._clock += n / self.hz
        return self.frames(n)

    def frames(self, n: int) -> List[FrameRecord]:
        """The next `n` frames, ignoring the clock (tests drive this)."""
        first = self.mock.frames
        by_frame: dict = {}
        for _side, lh in self.mock.generate(n):
            by_frame.setdefault(lh.frame_id, []).append(lh)
        out = []
        for k in range(first, first + n):
            t = k / self.hz
            env = self.scenario.environment(t)
            hands = by_frame.get(k, [])
            out.append(FrameRecord(
                t=t, frame_id=k, framerate=env["framerate"], hands=hands,
                image_mean=env["image_mean_hand"] if hands else env["image_mean"],
                image_saturated=env["image_saturated"], status=env["status"]))
        return out


# --- the run ----------------------------------------------------------------------
def scaled_steps(seconds: float):
    return [(start / 60.0 * seconds, label, text) for start, label, text in STEPS]


def beeper(enabled: bool):
    if not enabled:
        return None
    try:
        import winsound
        return AsyncBeeper(winsound.Beep)
    except ImportError:
        return None


def out_paths(out_dir: Path, stamp: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"tracking_quality_{stamp}"
    k = 2
    while base.with_suffix(".csv").exists() or base.with_suffix(".txt").exists():
        base = out_dir / f"tracking_quality_{stamp}_{k}"
        k += 1
    return base.with_suffix(".csv"), base.with_suffix(".txt")


def write_csv(path: Path, samples, prefer, t0) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SERIES_COLUMNS))
        writer.writeheader()
        writer.writerows(series_rows(samples, prefer, t0))


def print_plan(seconds: float, hand: Optional[str], mock: bool) -> None:
    who = f"bare {hand} hand" if hand else "bare hand"
    print("=" * 72)
    print(f"Tracking quality test: {seconds:g} s" + ("  [mock, no hardware]" if mock else ""))
    print(f"Setup: camera only, {who}. Module flat on the desk, lenses up.")
    print("Only one hand over the module; the other hand rests on the desk away from it.")
    print("Do this, slowly, and watch the camera window:")
    for start, label, text in scaled_steps(seconds):
        print(f"  {start:5.1f} s  {label:<20} {text}")
    print("A beep marks each step. Ctrl+C stops early and still writes the report.")
    print("=" * 72)


def run(args) -> int:
    mock = bool(args.mock)
    hand = args.hand
    print_plan(args.seconds, hand, mock)

    notes: List[str] = []
    if mock:
        source = MockSource(args.seconds, hand)
        source.start()
        notes.append("Mock run: a scripted hand, IR brightness and device status, no "
                     "camera. Nothing here is evidence about the real camera.")
    else:
        source = QualityStream(mode=args.mode)
        try:
            source.start()
        except LeapUnavailable as e:
            print(f"\nNo live tracking: {e}\n")
            return 2
        policy = source.enable_images()
        if policy:
            notes.append(f"Image policy: {policy}.")

    view = None
    if not mock and not args.no_view:
        view = CameraView(hand=hand, band=TARGET_CM).start()
    beep = beeper(not mock)
    hud = Hud(lambda s: print(s, end="", flush=True), every=HUD_EVERY_S)
    live = LiveLossTracker(prefer=hand)
    series: List[Sample] = []
    steps = scaled_steps(args.seconds)
    stopped_early = None
    t_run0 = None

    def caption(text: str) -> None:
        if view is not None:
            view.caption(text, band=TARGET_CM)

    try:
        if not mock:
            t_ready = time.time() + GET_READY_S
            print(f"\nGet ready: hand {TARGET_CM[0]:g} to {TARGET_CM[1]:g} cm over the "
                  "middle of the module, palm toward the lenses.")
            while time.time() < t_ready:
                left = t_ready - time.time()
                caption(f"GET READY: hand 30 cm over the module  {left:.0f}s")
                source.records()                     # keep the queue fresh
                time.sleep(0.02)
        if beep:
            beep.beep(1000, 250)
        t_run0 = time.time()
        step_shown = -1
        while True:
            now = time.time()
            elapsed = now - t_run0
            if elapsed >= args.seconds:
                break
            k = max(i for i, (start, _l, _t) in enumerate(steps) if start <= elapsed)
            new_step = k != step_shown
            if new_step:
                step_shown = k
                hud.close()
                print(f"\n>> {steps[k][1]}: {steps[k][2]}")
                if beep and k:
                    beep.beep(880, 150)
            for rec in source.records():
                sample = to_sample(rec)
                series.append(sample)
                live.add(sample)
            pct = 100.0 * (live.tracked_fraction or 0.0)
            last = "none" if live.last is None else cause_label(live.last.cause,
                                                                live.last.status)
            left = args.seconds - elapsed
            # The window's caption is a file the viewer re-reads, so it is
            # rewritten at the console line's pace, not on every pass.
            if hud.show(f"  {left:4.0f} s left   tracked {pct:3.0f} %   lost "
                        f"{live.total_losses}   last: {last}", now, force=new_step):
                caption(f"{steps[k][1]}  {left:.0f}s  tracked {pct:.0f}%  lost "
                        f"{live.total_losses}")
            time.sleep(0.005)
    except KeyboardInterrupt:
        stopped_early = time.time() - (t_run0 or time.time())
    finally:
        hud.close()
        if t_run0 is not None:                   # the frames of the last pass
            try:
                for rec in source.records():
                    series.append(to_sample(rec))
            except Exception:
                pass
        source.stop()
        if view is not None:
            view.close()
        if beep:
            beep.beep(500, 300)
            beep.stop()

    if stopped_early is not None:
        notes.append(f"Stopped early with Ctrl+C after {stopped_early:.1f} s.")
    if not mock:
        notes.extend(real_notes(source))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path, txt_path = out_paths(Path(args.out), stamp)
    t0 = series[0].t if series else 0.0
    report = summarise(series, prefer=hand, t_start=t0,
                       t_end=series[-1].t if series else None)
    who = f"bare {hand} hand" if hand else "bare hand"
    setup = (f"camera only, {who}; module flat on the desk, lenses up, desktop mode"
             + ("; MOCK camera" if mock
                else f"; device {source.device_serial or 'serial not read'}"))
    lines = format_report(report, title=f"Tracking quality test, {stamp}",
                          setup=setup, notes=notes)
    write_csv(csv_path, series, hand, t0)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPer-frame series: {csv_path}")
    print(f"Report:           {txt_path}\n")
    for line in lines:
        print(line)
    return 0


def real_notes(stream: QualityStream) -> List[str]:
    """What the camera could and could not tell this run, stated, not guessed."""
    notes = []
    if stream.images:
        notes.append(f"IR images: {stream.images} brightness readings "
                     f"({1 / IMAGE_EVERY_S:.0f} per second, left lens).")
    else:
        notes.append("IR images: none arrived, so image brightness is not available and "
                     "'bright background' could not be judged. Turn on 'Allow Images' "
                     "in the Ultraleap Control Panel and run it again.")
    if stream.status_events:
        raw = "-" if stream.status_raw is None else f"0x{stream.status_raw:08X}"
        notes.append(f"Device status: {stream.status_events} device event(s) from the "
                     f"service, last raw value {raw}, decoded here from the raw value "
                     "(the bindings' own decoding misreads plain streaming as bad "
                     "calibration and bad transport). LeapC has no low frame rate or "
                     "low USB bandwidth flag; the frame rate above is measured instead.")
    else:
        notes.append("Device status: the service sent no device event to this program, "
                     "so the smudged, robust (infrared interference) and low resource "
                     "flags could not be read.")
    notes.append(f"Dropped-frame events from the service: {stream.dropped_frame_events}.")
    return notes


def recompute(args) -> int:
    """`--from FILE`: the report again, from a saved CSV or a take's JSONL."""
    path = Path(args.from_file)
    if not path.is_file():
        print(f"no such file: {path}")
        return 2
    notes = [f"Recomputed from {path}."]
    gap_s = LOSS_GAP_S
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as fh:
            samples = samples_from_series_rows(csv.DictReader(fh))
    else:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        samples = rows_to_samples(rows)
        notes.append("A recording holds only frames with a hand in them: no image "
                     "brightness and no device status, so those causes are not judged.")
        gap_s = cadence_gap_s(samples, args.hand)
        if gap_s > LOSS_GAP_S:
            notes.append(f"This file was saved at a reduced rate, so only a hole longer "
                         f"than {gap_s * 1000:.0f} ms counts as a loss.")
    report = summarise(samples, prefer=args.hand, gap_s=gap_s)
    for line in format_report(report, title=f"Tracking quality, {path.name}",
                              notes=notes):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="The 60 second tracking test: camera only, bare hand. Explains "
                    "every time the camera loses the hand.")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="length of the coached run (default: 60)")
    p.add_argument("--hand", choices=("left", "right"), default=None,
                   help="the operator's hand; preferred when the tracker sees two "
                        "(its label is not trusted otherwise)")
    p.add_argument("--mock", action="store_true",
                   help="no hardware: a scripted mock hand, lost on purpose")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"folder for the CSV and the report (default: {DEFAULT_OUT})")
    p.add_argument("--mode", default="desktop", choices=("desktop", "hmd", "screentop"))
    p.add_argument("--from", dest="from_file", default=None,
                   help="re-read a saved tracking_quality CSV or a take's JSONL and "
                        "print the report, no camera")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.seconds <= 0:
        print("--seconds must be above 0")
        return 2
    if args.from_file:
        return recompute(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
