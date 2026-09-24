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

WHERE A LIVE DEMO'S DATA GOES
  Every --live run saves, in a folder made at the start,
  recordings/demo/<YYYY-MM-DD_HHMM>_<hands>/ (hands: left, right or both;
  _2, _3 and so on is added when a run in the same minute has the name):
    <hands>.jsonl          the fused frames, as fuse_live.py --out writes
                           them; with --hand both, both hands in this one
                           file, each line naming its hand
    <hands>.warmup.txt     what the console said up to the fusion
    <hands>.summary.txt    the end-of-run summary
    template_<hand>.json   each hand's bone measurement from the warm-up
                           (with --fit-template auto)
    demo.mp4               every frame the window drew while fusing, at 30
                           frames a second (the window draws 30 a second
                           while it records, so the video is real time).
                           OpenCV writes it with the mp4v codec; if ffmpeg
                           is on PATH it is then re-encoded to H.264 at CRF
                           23 and the mp4v file deleted. The summary says
                           which happened
    snapshot.png           the last frame drawn
    README.txt             every file with one line each, the hands, the
                           duration, the frame counts, and each hand's
                           paired share and camera use per DOF
  --out PATH.jsonl puts all of these beside PATH instead; --no-save writes
  none of them. When the run ends (q, Esc, --seconds, or Ctrl-C) the last
  line printed is "Data for this demo: FOLDER", and the folder opens in
  File Explorer (not with --no-open or --no-window).

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

A VIDEO OF A REPLAY
  --replay DIR --video PATH.mp4 draws without a window and writes every
  frame to an MP4 at 30 frames a second, the replay's real pace (times
  --speed): each video frame moves the replay on by 1/30 s, so the 60 Hz
  glove frames are sampled, not slowed down. Every take is played once, in
  the window's order; --takes N keeps only the first N. The frames are the
  window's own pictures, padded to an even width and height.

Flags shared with scripts/fuse_live.py (--hand, --profile, --glove-lag,
--fit-template, --warmup-*, --acquire-timeout, --max-dt, --drift-anchor,
--rail-fingers, --no-rail-override, --port, --mock-glove, --mock-leap,
--no-view, --out, --osc-out, --seconds) mean exactly what they mean there,
except that --out also moves the demo's other files (above); --out,
--osc-out and --seconds apply to --live only. The live setup below is
a copy of fuse_live.main's sequence (that function is one piece, so it
cannot be called part-way); every piece it calls is imported from there,
including what it says after the warm-up and at the end and the
PATH.warmup.txt and PATH.summary.txt it writes beside --out PATH.jsonl.

Usage:
  python scripts/demo.py --replay recordings/sync_day2
  python scripts/demo.py --live
  python scripts/demo.py --live --hand right
  python scripts/demo.py --live --no-save
  python scripts/demo.py --live --mock-glove --mock-leap --no-view
  python scripts/demo.py --replay recordings/sync_day2 --no-window \\
      --snapshot runs/demo.png --frames 400
  python scripts/demo.py --replay recordings/sync_day2 --video demo.mp4 \\
      --takes 12
"""
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
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
    FIT_FILE,
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
    canvas_size,
    even_size,
    pad_to,
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
    wall_clock,
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
# A --video file: frames a second, and the codecs tried, in order.
VIDEO_FPS = 30.0
VIDEO_CODECS = ("avc1", "mp4v")
# A live run's video: mp4v first, because OpenCV's avc1 on this laptop writes
# about 19 Mbit/s; ffmpeg, when it is on PATH, re-encodes it afterwards.
LIVE_VIDEO_CODECS = ("mp4v", "avc1")
H264_CRF = 23
# Where a live demo's data goes, and the names of the files in its folder.
DEMO_DIR = ROOT / "recordings" / "demo"
DEMO_VIDEO = "demo.mp4"
DEMO_SNAPSHOT = "snapshot.png"
DEMO_README = "README.txt"


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
        elif "--out" in action.option_strings:
            action.help = ("live: write the fused frames here and every "
                           "other file of the demo beside it, instead of in "
                           "recordings/demo/<YYYY-MM-DD_HHMM>_<hands>/")
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
    p.add_argument("--video", type=Path, default=None, metavar="PATH.mp4",
                   help="replay only: no window; play every take once and "
                        f"write each frame drawn to an MP4 at {VIDEO_FPS:g} "
                        "frames a second, at the replay's real pace "
                        "(times --speed)")
    p.add_argument("--takes", type=int, default=None, metavar="N",
                   help="replay only: play just the first N takes, in the "
                        "order the replay plays them")
    p.add_argument("--no-save", action="store_true",
                   help="live only: save nothing (by default a live run "
                        "saves its frames, logs, video, snapshot and a "
                        "README in recordings/demo/<YYYY-MM-DD_HHMM>_<hands>/)")
    p.add_argument("--no-open", action="store_true",
                   help="live only: do not open the data folder in File "
                        "Explorer at the end (--no-window never opens it)")
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


def write_snapshot(path: Optional[Path], img, say=print) -> Optional[Path]:
    """The last frame as a PNG (encoded here, so a path OpenCV's own writer
    cannot open still works). `say` is given the line naming the file."""
    if path is None or img is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"could not encode the snapshot for {path}")
    path.write_bytes(data.tobytes())
    say(f"Snapshot: {path}")
    return path


class VideoOut:
    """An MP4 file that the replay's frames are written to, one picture per
    frame, all padded to one even size (the codecs need even dimensions).

    The codecs are tried in the order given: by default avc1 (H.264) first,
    then mp4v. If this OpenCV can open none of them, it stops with a plain
    message.
    """

    def __init__(self, path: Path, size: Tuple[int, int],
                 fps: float = VIDEO_FPS, codecs: Sequence[str] = VIDEO_CODECS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = even_size(size)
        self.fps = float(fps)
        self.frames = 0
        self.writer = None
        self.codec = None
        for codec in codecs:
            writer = cv2.VideoWriter(str(self.path),
                                     cv2.VideoWriter_fourcc(*codec),
                                     self.fps, self.size)
            if writer.isOpened():
                self.writer, self.codec = writer, codec
                break
            writer.release()
        if self.writer is None:
            raise SystemExit(
                f"Could not write the video {self.path}: this OpenCV opens "
                f"neither the {' nor the '.join(codecs)} codec for an "
                "MP4 file.")

    def write(self, img) -> None:
        self.writer.write(pad_to(img, self.size))
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    @property
    def seconds(self) -> float:
        return self.frames / self.fps


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
    window = Window(enabled=not args.no_window and args.video is None)
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
    if args.takes is not None:
        takes = takes[:args.takes]
    clf, holdout, note = load_classifier(args.classifier_from, s,
                                         replay_dir=args.replay,
                                         replay_samples=samples)
    print(f"Playing {len(takes)} take(s)"
          + ("; " + REPLAY_KEYS if window.enabled else ""))
    player = ReplayPlayer(takes, args.speed)
    video = None
    if args.video is not None:
        rows = max(len([h for h in ("left", "right")
                        if h in t.hands and h in wanted]) for t in takes)
        video = VideoOut(args.video, canvas_size(rows))
        print(f"Writing {args.video} ({video.size[0]} x {video.size[1]}, "
              f"{VIDEO_FPS:g} fps, codec {video.codec}), every take once...")
    states: Dict[str, SideState] = {}
    drawn = 0
    last = time.perf_counter()
    try:
        while True:
            t_loop = time.perf_counter()
            if video is not None:
                # The replay's own clock moves 1/VIDEO_FPS s per video frame
                # (times the speed), however long the drawing takes.
                frames, changed = player.advance(
                    1.0 / VIDEO_FPS if drawn else 0.0)
                if changed:
                    if player.k == 0:
                        break                    # every take played once
                    # The new take's first frame, so no empty frame between.
                    frames, _ = player.advance(0.0)
            elif args.no_window:
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
            if video is not None:
                video.write(img)
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
        if video is not None:
            video.close()
    write_snapshot(args.snapshot, window.last)
    if video is not None:
        print(f"Video: {video.path} ({video.frames} frames, "
              f"{video.seconds:.1f} s)")
    print(f"Drew {drawn} frame(s).")
    return 0


# --- where a live demo's data goes -------------------------------------------

def demo_folder(hand: str, when: Optional[datetime] = None,
                base: Optional[Path] = None) -> Path:
    """A live demo's own folder, `recordings/demo/<YYYY-MM-DD_HHMM>_<hand>`
    (`base` in place of recordings/demo), `hand` being left, right or both.
    When a run earlier in the same minute already has that folder, `_2`,
    `_3` and so on is added, so no run overwrites another."""
    base = DEMO_DIR if base is None else Path(base)
    when = datetime.now() if when is None else when
    name = f"{when:%Y-%m-%d_%H%M}_{hand}"
    path, n = base / name, 2
    while path.exists():
        path, n = base / f"{name}_{n}", n + 1
    return path


def live_out_path(args, when: Optional[datetime] = None) -> Optional[Path]:
    """The fused frames' file of a live run, which every other file of the
    demo goes beside: --out PATH as given, else `<hand>.jsonl` in a new
    `demo_folder`; None with --no-save."""
    if args.no_save:
        return None
    if args.out is not None:
        return Path(args.out)
    return demo_folder(args.hand, when) / f"{args.hand}.jsonl"


def shown_path(path: Path, longest: int = 80) -> str:
    """`path` for the window's footer: relative to the working folder when
    it is inside it, else in full, or its last two parts after "..." when
    the full path is longer than `longest` characters."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        pass
    if len(str(path)) <= longest:
        return str(path)
    return os.sep.join(["...", path.parent.name, path.name])


def reencode_h264(path: Path, codec: str, crf: int = H264_CRF,
                  timeout: float = 900.0) -> str:
    """Re-encode the video at `path` to H.264 at `crf` with ffmpeg, in
    place, when ffmpeg is on PATH, and say in words what the file now is.
    The file OpenCV wrote (with `codec`) is replaced only by a finished
    re-encode; with no ffmpeg, or one that fails, it stays as it is."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return f"{codec} (ffmpeg is not on PATH, so not re-encoded)"
    tmp = path.with_name(path.stem + ".h264" + path.suffix)
    print(f"Re-encoding {path.name} to H.264 with ffmpeg...")
    try:
        done = subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(path),
             "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", str(tmp)],
            capture_output=True, text=True, timeout=timeout)
        ok = (done.returncode == 0 and tmp.is_file()
              and tmp.stat().st_size > 0)
        why = (done.stderr.strip().splitlines()
               or [f"exit code {done.returncode}"])[-1]
        if ok:
            os.replace(tmp, path)
    except (OSError, subprocess.SubprocessError) as e:
        ok, why = False, str(e)
    if not ok:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        return f"{codec} (the ffmpeg re-encode failed: {why})"
    return (f"H.264 CRF {crf}, re-encoded by ffmpeg from OpenCV's {codec} "
            "file (deleted)")


def finish_video(video: "VideoOut") -> Tuple[Optional[Path], str]:
    """Close a live run's video and re-encode it (`reencode_h264`).
    Returns (its path, what it is in words), or (None, why) when no frame
    was drawn and the empty file was removed."""
    video.close()
    if not video.frames:
        with suppress(OSError):
            video.path.unlink(missing_ok=True)
        return None, "no frame was drawn while fusing"
    return video.path, reencode_h264(video.path, video.codec)


def open_folder(folder: Path) -> None:
    """Show `folder` in File Explorer. Windows only: elsewhere, or if it
    fails, the run just ends."""
    startfile = getattr(os, "startfile", None)
    if startfile is None:
        return
    try:
        startfile(str(folder))
    except OSError as e:
        print(f"Could not open {folder}: {e}")


def summary_for_readme(lines: Sequence[str]) -> List[str]:
    """From `LiveFusion.summary_lines`: each hand's line (glove frames
    fused, the share paired with a camera frame, camera use, lag) and its
    camera use per gated DOF. Why the camera was refused and what the drift
    anchor did stay in the summary file."""
    out, keep = [], False
    for line in list(lines)[1:]:
        depth = len(line) - len(line.lstrip())
        text = line.strip()
        if depth <= 2:
            keep = not text.startswith("drift anchor")
        elif depth <= 4:
            keep = text.startswith("camera use per gated DOF")
        if keep:
            out.append(line)
    return out


def readme_lines(hands: Sequence[str], fused_hands: Sequence[str],
                 started: float, fusing_from: Optional[float], ended: float,
                 frames: str, files: Sequence[Tuple[Path, str]],
                 summary: Sequence[str] = ()) -> List[str]:
    """README.txt of a live demo's folder: when it ran, the hands, how long
    it fused, the frame counts, every file with one line on what it holds,
    and each hand's paired share and camera use per DOF from the summary."""
    asked = ", ".join(hands)
    shown = ", ".join(fused_hands) or "none"
    width = max((len(p.name) for p, _what in files), default=0) + 2
    lines = ["Glove + camera fusion demo (scripts/demo.py --live)",
             f"Run: {wall_clock(started)} to {wall_clock(ended)}",
             f"Hands: {shown}" + ("" if shown == asked else
                                  f" (asked for {asked}; the warm-up file "
                                  "says why)"),
             "Duration: " + ("nothing was fused" if fusing_from is None else
                             f"{ended - fusing_from:.1f} s of fusion")
             + f" ({ended - started:.1f} s with the warm-up)",
             f"Frames: {frames}",
             "",
             "Files:"]
    lines += [f"  {p.name:<{width}}{what}" for p, what in files]
    picked = summary_for_readme(summary)
    if picked:
        lines += ["", "From the summary (camera use per DOF counts paired "
                      "frames only):"] + picked
    return lines


# --- --live ------------------------------------------------------------------

def run_live(args, s: Settings) -> int:
    """fuse_live.main's sequence (profile, sensors, warm-up, fusion), with
    the demo window in place of nothing, and everything the run makes kept
    in one folder (see "WHERE A LIVE DEMO'S DATA GOES" above)."""
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

    out_path = live_out_path(args)
    folder = None if out_path is None else out_path.parent
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = RunLog(out_path, header=run_header(
        "demo.py --live", hands, s.profile_path, s.profile_how, lags,
        fit_spec))
    if folder is not None:
        log.say(f"Saving this demo in {folder}")
    listed: List[Tuple[Path, str]] = []

    def list_file(path, what) -> None:
        """Name `path` in README.txt, if it was written."""
        if path is not None and Path(path).is_file():
            listed.append((Path(path), what))

    def wrap_up(code, fused_hands=(), fusing_from=None, ended=None,
                frames="none fused", summary=()) -> int:
        """README.txt, then (after a fusion) the folder in File Explorer,
        and its path as the last line printed."""
        if folder is None:
            return code
        ended = time.time() if ended is None else ended
        if not listed:
            list_file(log.warmup_path, "what the console said up to the "
                      "fusion: settings, each hand's acquire, what the "
                      "warm-up learned")
        readme = folder / DEMO_README
        text = readme_lines(hands, fused_hands, started, fusing_from, ended,
                            frames, listed + [(readme, "this file")],
                            summary)
        try:
            readme.write_text("\n".join(text) + "\n", encoding="utf-8")
        except OSError as e:
            print(f"Could not write {readme}: {e}")
        # Opened whether or not fusing began: a run stopped with Ctrl-C during
        # the warm-up still leaves the warm-up report and this README, and
        # the question "where is the data" gets the same answer either way.
        if not (args.no_open or args.no_window):
            open_folder(folder)
        print(f"Data for this demo: {folder.resolve()}")
        return code
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
        return wrap_up(1)
    try:
        glove.start()
    except OSError as e:
        camera.stop()
        window.close()
        log.say(f"glove: cannot listen on OSC port {args.port} ({e}). Is "
                "another recorder still running?")
        log.write_warmup()
        return wrap_up(1)
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
        return wrap_up(1)
    except BaseException:
        shutdown()
        log.write_warmup()
        raise
    hud.close()
    camera.coach(None)
    buffer.keep_abs26 = False
    learned = warmup.learn(gates=gates, rail_params=rail_params, fit=fit)
    del warmup
    learned, fused_hands = conclude_warmup(learned, hands, fit, out_path,
                                           log)
    if not fused_hands:
        shutdown()
        return wrap_up(2)

    # --- the fusion (as fuse_live.main, drawing every frame) ---------------
    fusion = LiveFusion(learned, buffer, gates=gates, rail_params=rail_params,
                        unreliable=s.unreliable, anchor=s.anchor,
                        lag={h: lags[h][0] for h in fused_hands},
                        max_dt=args.max_dt)
    jsonl = None if out_path is None else JsonlSink(out_path)
    if jsonl is not None:
        sinks.append(jsonl)
    if osc_target is not None:
        sinks.append(OscSink(*osc_target))
    video = None
    if folder is not None:
        try:
            video = VideoOut(folder / DEMO_VIDEO, canvas_size(len(fused_hands)),
                             codecs=LIVE_VIDEO_CODECS)
        except SystemExit as e:              # the fusion goes on without it
            log.say(f"{e} Going on without a video.")
    # Recording, the window draws VIDEO_FPS times a second, so each picture
    # drawn is one video frame and the video plays in real time.
    period = FRAME_S if video is None else 1.0 / VIDEO_FPS
    footer = (LIVE_KEYS if folder is None
              else f"{LIVE_KEYS}   saving to {shown_path(folder)}")
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
                # A fixed schedule, so the draws keep their rate on average;
                # one that fell behind starts again from now.
                next_draw[0] = (now + period if now - next_draw[0] > period
                                else next_draw[0] + period)
                fresh_frames = False
                img = render(states, fused_hands, now, header=header,
                             footer=footer,
                             guesses=guesses_for(states, fused_hands, clf),
                             guess_note=note)
                drawn += 1
                if video is not None:
                    video.write(img)
                if window.show(img, 1) in ("q", "Q"):
                    break
                if args.frames is not None and drawn >= args.frames:
                    break
            else:
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        t_stop = time.time()
        shutdown()
        if video is not None:
            video.close()
    more = [f"Drew {drawn} frame(s)."]
    video_path, video_what = None, ""
    if video is not None:
        video_path, video_what = finish_video(video)
        more.append(f"Video: {video_path} ({video.frames} frames, "
                    f"{video.seconds:.1f} s at {video.fps:g} fps), "
                    f"{video_what}" if video_path is not None
                    else f"Video: none, {video_what}.")
    snapshot = (None if folder is None else
                write_snapshot(folder / DEMO_SNAPSHOT, window.last,
                               say=more.append))
    report_run(log, fusion, sinks, more=more)
    write_snapshot(args.snapshot, window.last)
    if folder is None:
        return 0
    list_file(out_path, "the fused frames, one JSON line per glove frame: "
              "stamps, hand, the 21 fused points, where each DOF came from"
              + ("; both hands in this one file" if len(hands) > 1 else ""))
    list_file(log.warmup_path, "what the console said up to the fusion: "
              "settings, each hand's acquire, what the warm-up learned")
    list_file(log.summary_path, "the end-of-run summary: pairing, camera use "
              "and refusals per DOF, the drift anchor, the outputs")
    if fit == FIT_AUTO:
        for hand in sorted(learned.measurements):
            list_file(folder / FIT_FILE.format(hand=hand),
                      f"the {hand} hand's bone measurement from the warm-up "
                      "(reuse it with --fit-template PATH)")
    list_file(video_path, f"every frame the window drew while fusing, "
              f"{VIDEO_FPS:g} fps, {video_what}")
    list_file(snapshot, "the last frame the window drew")
    frames = (f"{jsonl.written} fused frames saved, {drawn} drawn in the "
              "window, "
              + (f"{video.frames} in {DEMO_VIDEO}" if video_path is not None
                 else f"no {DEMO_VIDEO}"))
    return wrap_up(0, fused_hands, log.fusing_from, t_stop, frames,
                   fusion.summary_lines())


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if (args.replay is None) == (not args.live):
        print("Give exactly one of --replay DIR or --live.")
        return 1
    if args.speed <= 0:
        print("--speed must be above 0.")
        return 1
    if args.video is not None and args.replay is None:
        print("--video works with --replay only.")
        return 1
    if args.takes is not None and args.takes < 1:
        print("--takes must be 1 or more.")
        return 1
    if args.no_save and args.out is not None:
        print("--no-save and --out contradict each other: give one.")
        return 1
    s = settings_from_args(args)
    if args.replay is not None:
        return run_replay(args, s)
    return run_live(args, s)


if __name__ == "__main__":
    sys.exit(main())
