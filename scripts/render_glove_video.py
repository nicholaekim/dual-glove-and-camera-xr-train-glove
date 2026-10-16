"""Draw a glove session's takes as videos: one MP4 per accepted take.

No hardware. Sets B and C are recorded with the glove alone, so there is no
footage of the hand; this draws what the glove streamed instead
(`cam_hand.glove_video`): the hand at its palm and from its side, the cue in
force in words, the take's clock and each digit's flexion angles, in real
time. Every picture says along its bottom edge that it is drawn from the
glove's joint data and is not camera footage.

  python scripts/render_glove_video.py recordings/protocol/finger_flexion/20261013_180909_left
  python scripts/render_glove_video.py <session> --out "..\\xr trainer\\glove animations\\<session>"
  python scripts/render_glove_video.py <session> --items index,index_fast

The videos go to `<session>\\videos\\<take>.mp4` unless `--out` names another
folder. They never go into the hand-in folder: the packager refuses any
image or video there (`package_professor_set.IMAGE_EXTENSIONS`), so send
them beside it.

Only accepted takes are drawn (`--rejected` adds the rejected attempts, into
`rejected\\`). A take already drawn is drawn again and replaced. With ffmpeg
on PATH each file is re-encoded to H.264, which every player and browser
opens; without it the file stays OpenCV's mp4v.

Exit code 0 when every chosen take was drawn, 1 when one could not be, 2
when the folder is not a glove session.
"""
import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cam_hand import glove_video as gv  # noqa: E402
from cam_hand import protocol_check as pc  # noqa: E402

VIDEOS_DIR = "videos"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Draw each accepted take of a glove session (Set B or "
                    "Set C) as an MP4 of the hand the glove streamed.")
    p.add_argument("session", type=Path, help="the session folder")
    p.add_argument("--out", type=Path, default=None,
                   help="where the videos go (default: <session>\\videos)")
    p.add_argument("--items", default=None,
                   help="comma-separated item ids to draw (default: all)")
    p.add_argument("--fps", type=float, default=gv.FPS,
                   help=f"pictures per second (default {gv.FPS:g})")
    p.add_argument("--rejected", action="store_true",
                   help="draw the rejected attempts too, into rejected\\")
    p.add_argument("--no-h264", action="store_true",
                   help="keep OpenCV's mp4v file, no ffmpeg re-encode")
    args = p.parse_args(argv)
    if args.fps <= 0:
        p.error("--fps must be above 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        session = pc.load_session(args.session)
    except (FileNotFoundError, NotADirectoryError):
        print(f"No session.json in {args.session}")
        return 2
    if session.set not in (pc.FLEXION, pc.SEQUENCES):
        print(f"{args.session} is a {session.set} session: only the glove "
              "sets (finger_flexion, sequences) are drawn")
        return 2
    wanted = ({x.strip() for x in args.items.split(",") if x.strip()}
              if args.items else None)
    takes = [t for t in session.takes
             if (t.accepted or args.rejected)
             and (wanted is None or t.item in wanted)]
    if wanted:
        unknown = wanted - {t.item for t in session.takes}
        if unknown:
            print("No take of: " + ", ".join(sorted(unknown)))
            return 1
    if not takes:
        print("No take to draw.")
        return 1
    out = args.out or (session.path / VIDEOS_DIR)
    mock = pc.mock_flags(session.meta)
    print(f"{session.set}, {session.hand} hand, session {session.name}: "
          f"{len(takes)} take(s) -> {out}")
    failed = 0
    for t in takes:
        glove = t.path("glove")
        if glove is None:
            print(f"  {t.name}: no glove file, not drawn")
            failed += 1
            continue
        dest = (out if t.accepted else out / "rejected") / f"{t.name}.mp4"
        title = (f"{t.name}   ({session.set}, {session.hand} hand"
                 + (", MOCK data" if mock else "")
                 + ("" if t.accepted else ", REJECTED attempt") + ")")
        began = time.time()
        try:
            info = gv.render_take(dest, glove, pc.read_events(t.path("events")),
                                  session.hand, title, fps=args.fps,
                                  h264=not args.no_h264)
        except (OSError, ValueError) as e:
            print(f"  {t.name}: NOT DRAWN: {e}")
            failed += 1
            continue
        print(f"  {dest.name}: {info['seconds']:.1f} s, {info['pictures']} "
              f"pictures at {info['fps']:g}/s from {info['glove_frames']} "
              f"glove frames, {info['codec']} "
              f"({time.time() - began:.0f} s to draw)")
    print(f"{len(takes) - failed} of {len(takes)} take(s) drawn.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
