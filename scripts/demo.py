"""Watch the fusion: the glove, camera and fused hands side by side, moving.

One window, one row per hand. Each row shows three hands drawn the same way
and at the same scale (see `cam_hand.demo_view`):

  GLOVE  (orange)  the glove's hand as it went into the fusion (the fitted
                   template hand when a template fit is in force)
  CAMERA (blue)    the Ultraleap frame the fusion paired with it
  FUSED  (green)   what `LiveFusion.step` made of the two

Under the fused hand, one line per finger says where that finger's curl and
spread came from in this frame (glove or camera), where the thumb's direction
came from, which fingers the rail override is holding (the camera taking a
curl the glove cannot report, e.g. on a pinch), whether the camera is fresh,
and the glove's rate. Under the glove and camera hands, a pose guess: the
report's nearest-centroid classifier on the fused hand's features, with the
margin to the next pose. A stream not seen for half a second goes grey and
says "no hand".

TWO MODES, THE SAME WINDOW

  --replay DIR   no hardware. The recorded session (DIR/glove, DIR/leap, as
                 written by record_simultaneous.py) is read the way
                 fuse_poses.py reads it, learned from as `fuse_live.py
                 --replay` learns from it, fused take by take through the
                 live path (`replay_take`, i.e. `LiveFusion.step`), and then
                 played at the recording's own pace, take after take, with
                 the take's pose shown beside the guess. It loops at the end.
                 Takes are played take 1 of every pose, then take 2, and so
                 on. Reading and fusing sync_day2 takes about half a minute.
                 Keys: space pause, n next take, r restart this take,
                 + / - speed, q (or Esc) quit.

  --live         the gloves and the Ultraleap. The same profile, lag,
                 ACQUIRE gate, warm-up (open palm, then fist, per hand) and
                 fusion as scripts/fuse_live.py, with every fused frame drawn
                 as it is made. q quits. The warm-up's instructions are in
                 this window, on the console, and in the camera window
                 (--no-view closes that one).

THE POSE GUESS
  Centroids are the per-take means of the fused hand's `all_features` over
  the session given by --classifier-from (default: the replayed session in
  a replay; recordings/sync_day2 live, if it exists; `none` turns the guess
  off). Replaying the session the centroids came from, the take on screen
  is left out of them (leave-one-take-out, as in the report), so a take is
  never guessed by centroids it helped build.

HEADLESS (for tests and a look without a screen)
  --no-window --snapshot PATH --frames N renders N frames without a window
  (a replay then runs as fast as it can) and writes the last one as a PNG.

Flags shared with scripts/fuse_live.py (--hand, --profile, --glove-lag,
--fit-template, --warmup-*, --acquire-timeout, --max-dt, --drift-anchor,
--rail-fingers, --no-rail-override, --port, --mock-glove, --mock-leap,
--no-view, --out, --osc-out, --seconds) mean exactly what they mean there;
--out, --osc-out and --seconds apply to --live only. The live setup below is
a copy of fuse_live.main's sequence (that function is one piece, so it
cannot be called part-way); every piece it calls is imported from there,
including what it says after the warm-up and at the end and the
PATH.warmup.txt and PATH.summary.txt it writes beside --out PATH.jsonl.

Usage:
  python scripts/demo.py --replay recordings/sync_day2
  python scripts/demo.py --live
  python scripts/demo.py --live --hand right
  python scripts/demo.py --live --mock-glove --mock-leap --no-view
  python scripts/demo.py --replay recordings/sync_day2 --no-window \\
      --snapshot runs/demo.png --frames 400
"""
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

import cv2  # noqa: E402

import fuse_live  # noqa: E402
from fuse_live import (  # noqa: E402
    ACQUIRED_FREQ,
    CUE_FREQ,
    CUE_MS,
    acquired_line,
    conclude_warmup,
    report_run,
    resolve_lag,
    run_header,
)
from fuse_poses import (  # noqa: E402
    AUTO,
    CAM_DIRS,
    FIT_AUTO,
    FIT_NONE,
    LEAP,
    fit_measurements,
    glove_lag_of,
    load_profile,
    parse_glove_lag,
    parse_rail_fingers,
    read_session,
    resolve_profile,
)

from cam_hand.demo_view import (  # noqa: E402
    CentroidClassifier,
    SideState,
    pose_features,
    render,
    render_message,
)
from cam_hand.features import flexion_features, mean_vector  # noqa: E402
from cam_hand.fusion import (  # noqa: E402
    DEFAULT_GATES,
    DEFAULT_RAIL,
    DriftAnchorParams,
    RailOverrideParams,
)
from cam_hand.live_fusion import (  # noqa: E402
    COUNTDOWN_S,
    MOCK_POSE,
    STAGE_ACQUIRE,
    STAGE_COUNTDOWN,
    STAGE_NOT_ACQUIRED,
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
from cam_hand.template_fit import load_measurement  # noqa: E402
from leap_hand.live import beep  # noqa: E402
from leap_hand.protocol import (  # noqa: E402
    DEFAULT_BAND,
    HUD_EVERY,
    AsyncBeeper,
    CameraView,
    Hud,
)

WINDOW = "Glove + camera fusion demo"
DEFAULT_CLASSIFIER_DIR = ROOT / "recordings" / "sync_day2"
CLASSIFIER_AUTO = "auto"
CLASSIFIER_NONE = "none"
# A replay holds the last frame of a take this long before the next take.
TAKE_GAP_S = 0.4
MIN_SPEED, MAX_SPEED = 0.125, 8.0
# The longest wall-clock step a replay advances by (a dragged window must not
# make it jump seconds ahead).
MAX_STEP_S = 0.25
FRAME_S = 1.0 / 60.0
REPLAY_KEYS = ("space pause   n next take   r restart take   + / - speed   "
               "q quit")
LIVE_KEYS = "q quit"


# --- the command line --------------------------------------------------------

def build_parser():
    """scripts/fuse_live.py's parser, so the shared flags mean the same
    thing, with this command's own flags added."""
    p = fuse_live.build_parser()
    p.description = __doc__.split("\n", 1)[0]
    p.epilog = __doc__.split("\n", 1)[1]
    for action in p._actions:
        if "--replay" in action.option_strings:
            action.help = ("no hardware: play a recorded session (DIR/glove, "
                           "DIR/leap) through the live fusion path at the "
                           "recording's pace, take after take")
    p.add_argument("--live", action="store_true",
                   help="the gloves and the Ultraleap: warm up, then draw "
                        "every fused frame as it is made")
    p.add_argument("--speed", type=float, default=1.0,
                   help="replay speed, 1 = real time (default 1; + and - "
                        "double and halve it while playing)")
    p.add_argument("--classifier-from", default=CLASSIFIER_AUTO,
                   metavar="{auto,none,DIR}",
                   help="session whose per-take fused features are the pose "
                        "guess's centroids. Default auto: the replayed "
                        "session, or live recordings/sync_day2 if it exists")
    p.add_argument("--no-window", action="store_true",
                   help="draw without opening a window (with --snapshot)")
    p.add_argument("--snapshot", type=Path, default=None, metavar="PATH.png",
                   help="write the last frame drawn as a PNG")
    p.add_argument("--frames", type=int, default=None,
                   help="stop after drawing this many frames")
    return p


@dataclass
class Settings:
    """What a fusion needs, read from the flags as fuse_live.main reads
    them."""
    gates: object
    rail_params: Optional[RailOverrideParams]
    unreliable: dict
    profile_path: Optional[Path]
    profile_how: str
    profile_name: str
    profile_lag: dict
    anchor: Optional[DriftAnchorParams]
    fit_spec: str
    lag_text: str
    max_dt: float


def settings_from_args(args) -> Settings:
    """The profile, the masks, the rail fingers and the drift anchor, with
    fuse_live.main's precedence: a flag beats the profile, the profile beats
    the default."""
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
    anchor = (DriftAnchorParams(window_s=args.anchor_window,
                                hold_s=args.anchor_hold,
                                deadband=args.anchor_deadband)
              if args.drift_anchor == "on" else None)
    return Settings(gates=DEFAULT_GATES, rail_params=rail_params,
                    unreliable=unreliable, profile_path=profile_path,
                    profile_how=profile_how, profile_name=profile_name,
                    profile_lag=profile_lag, anchor=anchor,
                    fit_spec=str(args.fit_template).strip(),
                    lag_text=args.glove_lag, max_dt=args.max_dt)


# --- a recorded session, fused through the live path -------------------------

@dataclass
class Take:
    """One recorded take, fused: its frames in stamp order."""
    name: str
    pose: str
    number: int
    hands: Tuple[str, ...]
    frames: list


def _take_number(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fuse_session(input_dir, s: Settings, log=print):
    """Read a session as fuse_poses.py does and fuse every take through the
    live path, as `fuse_live.py --replay` does.

    Returns `(takes, samples)`: the fused takes, and per take and hand the
    mean `all_features` of the fused hand as `(pose, hand, take, features)`,
    the shape of `fuse_poses.FusionRun.fused_samples`.
    """
    input_dir = Path(input_dir)
    glove_dir = input_dir / "glove"
    cam_dirs = [input_dir / d for d in CAM_DIRS if (input_dir / d).is_dir()]
    if not glove_dir.is_dir() or not cam_dirs:
        raise SystemExit(f"{input_dir}: expected {glove_dir} and "
                         f"{input_dir / LEAP}")
    fit_auto = s.fit_spec.lower() == FIT_AUTO
    loaded, skipped, _by_source, _by_clock, measure_parts = read_session(
        input_dir, take_files(glove_dir), AUTO, s.gates, measure=fit_auto)
    skipped += [(e["name"], "MediaPipe take: live fusion is Ultraleap only")
                for e in loaded if e["source"] != LEAP]
    loaded = [e for e in loaded if e["source"] == LEAP]
    for name, why in skipped:
        log(f"  skipped {name}: {why}")
    if not loaded:
        raise SystemExit(f"{input_dir}: no Ultraleap take to play")
    lag_rows, glove_lag = glove_lag_of(parse_glove_lag(s.lag_text), input_dir,
                                       cam_dirs, loaded,
                                       profile_lag=s.profile_lag)
    measurements, _scales, _saved, refusals = fit_measurements(
        s.fit_spec, measure_parts, loaded, input_dir, save=False)
    for hand, row in sorted(lag_rows.items()):
        log(f"  {hand}: glove lag "
            + (f"{row.seconds:.3f} s ({row.source})" if row.applied
               else "none applied")
            + ("; template fitted" if hand in measurements else "")
            + (f"; template fit refused ({refusals[hand]})"
               if hand in refusals else ""))
    learned = learn_from_session(
        [g for e in loaded for g in e["glove"]],
        [c for e in loaded for c in e["cam"]],
        gates=s.gates, rail_params=s.rail_params, measurements=measurements)
    takes, samples = [], []
    for entry in loaded:
        frames = replay_take(entry["glove"], entry["cam"], learned,
                             gates=s.gates, rail_params=s.rail_params,
                             unreliable=s.unreliable, anchor=s.anchor,
                             lag=glove_lag or {}, max_dt=s.max_dt,
                             clock=entry["clock"])
        if not frames:
            continue
        first = entry["glove"][0]
        pose = str(first.get("pose", "")) or "?"
        per_hand = defaultdict(list)
        for f in frames:
            per_hand[f.hand].append(pose_features(f.pts, f.hand))
        for hand, feats in per_hand.items():
            samples.append((pose, hand, entry["name"], mean_vector(feats)))
        takes.append(Take(entry["name"], pose, _take_number(first.get("take")),
                          tuple(sorted(per_hand)), frames))
    return takes, samples


def play_order(takes: Sequence[Take]) -> List[Take]:
    """Take 1 of every pose, then take 2, and so on: a demo shows every pose
    early instead of five fists in a row."""
    return sorted(takes, key=lambda t: (t.number, t.name))


def load_classifier(spec: str, s: Settings, replay_dir=None,
                    replay_samples=None, log=print
                    ) -> Tuple[Optional[CentroidClassifier], bool, str]:
    """(classifier or None, leave the current take out?, a note to show)."""
    spec = str(spec).strip()
    if spec.lower() == CLASSIFIER_NONE:
        return None, False, "off (--classifier-from none)"
    if spec.lower() == CLASSIFIER_AUTO:
        if replay_dir is not None:
            path = Path(replay_dir)
        elif DEFAULT_CLASSIFIER_DIR.is_dir():
            path = DEFAULT_CLASSIFIER_DIR
        else:
            return None, False, f"off (no {DEFAULT_CLASSIFIER_DIR.name})"
    else:
        path = Path(spec)
    same = (replay_dir is not None
            and path.resolve() == Path(replay_dir).resolve())
    if same:
        samples = replay_samples
    else:
        log(f"Pose centroids: fusing {path} (this takes a while)...")
        _takes, samples = fuse_session(path, s, log=log)
    clf = CentroidClassifier(samples, source=path.name)
    if len(clf.poses) < 2:
        return None, False, f"off ({path.name} has fewer than two poses)"
    note = (f"centroids: {path.name}, this take left out" if same
            else f"centroids: {path.name}")
    log(f"Pose centroids from {path}: {', '.join(clf.poses)}")
    return clf, same, note


# --- the window --------------------------------------------------------------

def screen_size() -> Optional[Tuple[int, int]]:
    """The primary screen's (width, height), or None off Windows."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return None


class Window:
    """The OpenCV window, or nothing at all with --no-window. The picture
    is drawn at one size; a window taller than the screen is shown scaled
    down to fit (and can be resized by dragging)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.size = None
        self.last = None

    def show(self, img, wait_ms: int = 1) -> Optional[str]:
        """Show `img`; the key pressed, 'q' when the window was closed."""
        self.last = img
        if not self.enabled:
            return None
        if self.size is None:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if self.size != img.shape[:2]:
            self.size = img.shape[:2]
            h, w = self.size
            screen = screen_size()
            f = (1.0 if screen is None
                 else min(1.0, 0.95 * screen[0] / w, 0.88 * screen[1] / h))
            cv2.resizeWindow(WINDOW, int(w * f), int(h * f))
        cv2.imshow(WINDOW, img)
        code = cv2.waitKey(max(1, int(wait_ms)))
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return "q"
        except cv2.error:
            return "q"
        if code < 0:
            return None
        code &= 0xFF
        return "q" if code == 27 else chr(code)

    def close(self) -> None:
        if self.enabled and self.size is not None:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass


def write_snapshot(path: Optional[Path], img) -> Optional[Path]:
    """The last frame as a PNG (encoded here, so a path OpenCV's own writer
    cannot open still works)."""
    if path is None or img is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"could not encode the snapshot for {path}")
    path.write_bytes(data.tobytes())
    print(f"Snapshot: {path}")
    return path


def guesses_for(states: Dict[str, SideState], sides: Sequence[str],
                clf: Optional[CentroidClassifier],
                exclude_take: Optional[str] = None) -> dict:
    if clf is None:
        return {}
    return {side: clf.guess(states[side].mean_features(), exclude_take)
            for side in sides if side in states}


# --- --replay ----------------------------------------------------------------

class ReplayPlayer:
    """Fused takes played on their own clock: take after take, looping.

    `clock` is the current take's time (its glove stamps). `advance` moves it
    by a wall-clock step times the speed and returns the frames it passed;
    `step` returns the next frame whatever its stamp (headless).
    """

    def __init__(self, takes: Sequence[Take], speed: float = 1.0):
        if not takes:
            raise ValueError("nothing to play")
        self.takes = list(takes)
        self.speed = min(MAX_SPEED, max(MIN_SPEED, float(speed)))
        self.paused = False
        self.start(0)

    @property
    def take(self) -> Take:
        return self.takes[self.k]

    def start(self, k: int) -> None:
        self.k = k % len(self.takes)
        self.i = 0
        self.clock = self.take.frames[0].t_glove

    def next_take(self) -> None:
        self.start(self.k + 1)

    def restart(self) -> None:
        self.start(self.k)

    def faster(self) -> None:
        self.speed = min(MAX_SPEED, self.speed * 2.0)

    def slower(self) -> None:
        self.speed = max(MIN_SPEED, self.speed / 2.0)

    @property
    def progress(self) -> float:
        frames = self.take.frames
        span = frames[-1].t_glove - frames[0].t_glove
        return 1.0 if span <= 0 else (self.clock - frames[0].t_glove) / span

    def advance(self, dt: float) -> Tuple[list, bool]:
        """(frames passed, did the take change). On a change no frame is
        returned: any passed in the same step were the old take's."""
        if self.paused:
            return [], False
        self.clock += min(dt, MAX_STEP_S) * self.speed
        frames = self.take.frames
        out = []
        while self.i < len(frames) and frames[self.i].t_glove <= self.clock:
            out.append(frames[self.i])
            self.i += 1
        if (self.i >= len(frames)
                and self.clock >= frames[-1].t_glove + TAKE_GAP_S):
            self.next_take()
            return [], True
        return out, False

    def step(self) -> Tuple[list, bool]:
        """The next frame, as fast as asked."""
        changed = False
        if self.i >= len(self.take.frames):
            self.next_take()
            changed = True
        f = self.take.frames[self.i]
        self.i += 1
        self.clock = f.t_glove
        return [f], changed


def run_replay(args, s: Settings) -> int:
    wanted = ("left", "right") if args.hand == "both" else (args.hand,)
    # The window opens once the session is fused: a window left without
    # its message loop for half a minute would be marked "not responding".
    window = Window(enabled=not args.no_window)
    print(f"Replay of {args.replay} (reading and fusing every take first): "
          "profile "
          f"{s.profile_path.name if s.profile_path else '(none)'}, "
          f"template fit {s.fit_spec}, drift anchor "
          f"{'on' if s.anchor is not None else 'off'}")
    takes, samples = fuse_session(args.replay, s)
    takes = [t for t in play_order(takes) if set(t.hands) & set(wanted)]
    if not takes:
        print(f"--replay {args.replay}: no take of the {args.hand} hand")
        return 1
    clf, holdout, note = load_classifier(args.classifier_from, s,
                                         replay_dir=args.replay,
                                         replay_samples=samples)
    print(f"Playing {len(takes)} take(s)"
          + ("" if args.no_window else "; " + REPLAY_KEYS))
    player = ReplayPlayer(takes, args.speed)
    states: Dict[str, SideState] = {}
    drawn = 0
    last = time.perf_counter()
    try:
        while True:
            t_loop = time.perf_counter()
            if args.no_window:
                frames, changed = player.step()
            else:
                frames, changed = player.advance(t_loop - last)
            last = t_loop
            if changed:
                states = {}
            for f in frames:
                states.setdefault(f.hand, SideState(f.hand)).update(f)
            take = player.take
            sides = [h for h in ("left", "right")
                     if h in take.hands and h in wanted]
            header = [f"REPLAY {Path(args.replay).name}   take {player.k + 1}"
                      f"/{len(takes)}   {take.name}   speed x{player.speed:g}"
                      + ("   PAUSED" if player.paused else ""),
                      f"pose: {take.pose}"]
            img = render(states, sides, player.clock, header=header,
                         footer=REPLAY_KEYS,
                         guesses=guesses_for(states, sides, clf,
                                             take.name if holdout else None),
                         truth=take.pose, guess_note=note,
                         progress=player.progress)
            drawn += 1
            spent = time.perf_counter() - t_loop
            key = window.show(img, int(1000 * max(0.0, FRAME_S - spent)))
            if key in ("q", "Q"):
                break
            if key == " ":
                player.paused = not player.paused
            elif key in ("n", "N"):
                player.next_take()
                states = {}
            elif key in ("r", "R"):
                player.restart()
                states = {}
            elif key in ("+", "="):
                player.faster()
            elif key in ("-", "_"):
                player.slower()
            if args.frames is not None and drawn >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        window.close()
    write_snapshot(args.snapshot, window.last)
    print(f"Drew {drawn} frame(s).")
    return 0


# --- --live ------------------------------------------------------------------

def run_live(args, s: Settings) -> int:
    """fuse_live.main's sequence (profile, sensors, warm-up, fusion), with
    the demo window in place of nothing."""
    hands = ("left", "right") if args.hand == "both" else (args.hand,)
    gates, rail_params = s.gates, s.rail_params
    lags = resolve_lag(parse_glove_lag(args.glove_lag), hands, s.profile_lag,
                       s.profile_path)
    fit_spec = s.fit_spec
    if fit_spec.lower() in (FIT_AUTO, FIT_NONE):
        fit = fit_spec.lower()
    else:
        try:
            fit = load_measurement(fit_spec)
        except (OSError, ValueError) as e:
            raise SystemExit(f"--fit-template {fit_spec}: {e}")
    osc_target = (fuse_live.parse_host_port(args.osc_out)
                  if args.osc_out is not None else None)

    window = Window(enabled=not args.no_window)
    clf, _holdout, note = load_classifier(args.classifier_from, s)

    log = RunLog(args.out, header=run_header(
        "demo.py --live", hands, s.profile_path, s.profile_how, lags,
        fit_spec))
    log.say(f"Profile: {s.profile_path or '(none)'}  [{s.profile_how}]"
            + (f"  {s.profile_name}" if s.profile_name else ""))
    for hand in hands:
        seconds, why = lags[hand]
        masked = ", ".join(s.unreliable.get(hand, ())) or "(none)"
        override = ("off (--no-rail-override)" if rail_params is None
                    else ", ".join(rail_params.fingers_for(hand)) or "(none)")
        log.say(f"  {hand}: glove lag {seconds:.3f} s ({why}); unreliable "
                f"{masked}; rail override {override}")
    log.say(f"Template fit: {fit_spec}.  Drift anchor: {args.drift_anchor}.  "
            f"Pairing within {args.max_dt:g} s.")

    # --- the sensors (as fuse_live.main) ------------------------------------
    buffer = CameraBuffer(gates=gates, keep_abs26=True)
    camera = CameraSource(hands, mock=args.mock_leap)
    glove = GloveSource(hands, mock=args.mock_glove, port=args.port)
    try:
        camera.start()
    except Exception as e:                  # LeapUnavailable says what to do
        log.say(f"camera: {e}")
        log.write_warmup()
        window.close()
        return 1
    try:
        glove.start()
    except OSError as e:
        camera.stop()
        window.close()
        log.say(f"glove: cannot listen on OSC port {args.port} ({e}). Is "
                "another recorder still running?")
        log.write_warmup()
        return 1
    view = CameraView(hand=args.hand if args.hand != "both" else None,
                      band=DEFAULT_BAND,
                      enabled=not (args.no_view or args.mock_leap)).start()
    beeper = AsyncBeeper(beep)
    hud = Hud(lambda text: print(text, end="", flush=True), every=HUD_EVERY)
    sinks = []

    def shutdown():
        hud.close()
        for closer in (glove.stop, camera.stop, view.close, beeper.stop,
                       window.close, *[k.close for k in sinks]):
            try:
                closer()
            except Exception:
                pass

    # --- the warm-up (as fuse_live.main, plus this window) -----------------
    warmup = Warmup(hands, open_s=args.warmup_open, fist_s=args.warmup_fist,
                    acquire_s=args.acquire_timeout, countdown_s=COUNTDOWN_S,
                    band=DEFAULT_BAND)
    words = band_words(DEFAULT_BAND)
    log.say("Warm-up, one hand at a time ("
            + ", then ".join(h.upper() for h in hands)
            + f"): acquire, open palm {args.warmup_open:g} s, fist "
            f"{args.warmup_fist:g} s.")
    shown = [None]
    status = [""]
    next_hud = [0.0]
    next_draw = [0.0]
    asked = {}

    def caption(text):
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

    def draw_warmup():
        now = time.time()
        if now < next_draw[0]:
            return
        next_draw[0] = now + 1.0 / 15.0
        key = window.show(render_message(
            [f"WARM-UP  {shown[0] or ''}", status[0],
             "Follow the beeps: ACQUIRE (open hand over the module, palm to "
             "the lens),", "then OPEN PALM flat to the camera, then a FIST.",
             "q quits"], len(hands)))
        if key in ("q", "Q"):
            raise KeyboardInterrupt

    def poll(w):
        for row in camera.drain():
            w.add_camera(buffer.add(row))
        for g in glove.drain():
            w.add_glove(g["hand_side"], flexion_features(g["pts"]),
                        g["frame"])
        draw_warmup()

    def acquire(hand, now):
        other = "right" if hand == "left" else "left"
        return acquire_status(hand, glove.recent_packets(hand, now),
                              glove.recent_packets(other, now),
                              camera.reading(hand, now),
                              camera.seen_recently(other, now),
                              band=DEFAULT_BAND)

    def show(stage, hand, left, st):
        now = time.time()
        if now < next_hud[0]:
            return
        next_hud[0] = now + HUD_EVERY
        if stage == STAGE_ACQUIRE:
            line = st.line(left)
            caption(f"{hand.upper()} hand: ACQUIRE, {st.caption_text}")
        elif stage == STAGE_COUNTDOWN:
            line = countdown_text(hand, left)
            caption(line)
        else:
            hz = glove.rate_hz(hand)
            fresh = camera_fresh(buffer, hand, now)
            line = (f"{hand.upper()} {stage[:9]:<9} {max(0.0, left):4.1f}s  "
                    f"glove {'--' if hz is None else f'{hz:.0f}'}/s  "
                    f"camera {'fresh' if fresh else 'STALE'}")
        status[0] = line
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

    # --- the fusion (as fuse_live.main, drawing every frame) ---------------
    fusion = LiveFusion(learned, buffer, gates=gates, rail_params=rail_params,
                        unreliable=s.unreliable, anchor=s.anchor,
                        lag={h: lags[h][0] for h in fused_hands},
                        max_dt=args.max_dt)
    if args.out is not None:
        sinks.append(JsonlSink(args.out))
    if osc_target is not None:
        sinks.append(OscSink(*osc_target))
    beeper.beep(CUE_FREQ, CUE_MS)
    view.caption("LIVE FUSION", band=DEFAULT_BAND)
    print("Fusing" + (f" for {args.seconds:g} s" if args.seconds else
                      " until q or Ctrl-C") + ".")
    log.fusing()
    latest = {}
    states: Dict[str, SideState] = {}
    header = [f"LIVE   {' + '.join(h.upper() for h in fused_hands)}   "
              f"profile {s.profile_path.name if s.profile_path else '(none)'}"
              "   lag " + "  ".join(f"{h[0].upper()} {lags[h][0]:.2f} s"
                                   for h in fused_hands),
              f"template fit {fit_spec}"]
    drawn = 0
    fresh_frames = False
    t_end = None if args.seconds is None else time.time() + args.seconds
    try:
        while t_end is None or time.time() < t_end:
            for row in camera.drain():
                buffer.add(row)
            for g in glove.drain():
                out = fusion.step(g)
                if out is None:
                    continue
                latest[out.hand] = out
                states.setdefault(out.hand, SideState(out.hand)).update(out)
                fresh_frames = True
                for sink in sinks:
                    sink.write(out)
            now = time.time()
            if now >= next_hud[0]:
                next_hud[0] = now + HUD_EVERY
                hud.show(hud_line([hud_segment(h, latest.get(h),
                                               glove.rate_hz(h),
                                               camera_fresh(buffer, h, now))
                                   for h in fused_hands]), now, force=True)
            # Headless, every batch of new frames is drawn; with a window, at
            # most 60 times a second.
            due = (fresh_frames if args.no_window
                   else now >= next_draw[0])
            if due:
                next_draw[0] = now + FRAME_S
                fresh_frames = False
                img = render(states, fused_hands, now, header=header,
                             footer=LIVE_KEYS,
                             guesses=guesses_for(states, fused_hands, clf),
                             guess_note=note)
                drawn += 1
                if window.show(img, 1) in ("q", "Q"):
                    break
                if args.frames is not None and drawn >= args.frames:
                    break
            else:
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    report_run(log, fusion, sinks, more=[f"Drew {drawn} frame(s)."])
    write_snapshot(args.snapshot, window.last)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if (args.replay is None) == (not args.live):
        print("Give exactly one of --replay DIR or --live.")
        return 1
    if args.speed <= 0:
        print("--speed must be above 0.")
        return 1
    s = settings_from_args(args)
    if args.replay is not None:
        return run_replay(args, s)
    return run_live(args, s)


if __name__ == "__main__":
    sys.exit(main())
