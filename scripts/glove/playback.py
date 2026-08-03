"""Open a recorded session and scrub through it like a video.

Usage:
    python scripts/playback.py recordings/session_w9que9qu.jsonl

Controls:
    - drag the "Frame" slider to scrub back and forth
    - left / right arrow keys step one frame at a time
    - Play / Pause button auto-advances at the recorded rate

The panel on the right shows every joint's position for the frame you're on,
and highlights the joint that moved the most into that frame.
"""
import argparse
import logging
from pathlib import Path

from xr_hand.playback_viewer import PlaybackViewer
from xr_hand.recorder import FrameRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("playback")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="Path to a recorded .jsonl session.")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"file not found: {args.input}")

    frames_with_time = list(FrameRecorder.load(args.input))
    if not frames_with_time:
        raise SystemExit("recording is empty")

    log.info("loaded %d frames from %s", len(frames_with_time), args.input)
    viewer = PlaybackViewer(
        frames_with_time,
        title=f"Playback: {args.input.name}",
        source_path=args.input,
    )
    viewer.show()


if __name__ == "__main__":
    main()
