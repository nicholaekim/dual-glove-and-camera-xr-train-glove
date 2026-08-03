"""Real OSC pipeline: receiver -> validator -> parser -> 3D viewer.

Usage:
    python scripts/run_osc.py                         # defaults
    python scripts/run_osc.py --port 9000
    python scripts/run_osc.py --dump --no-viz         # diagnose stream only

When first connecting XR Trainer, run with --dump --no-viz to confirm the
address pattern and arg count match what the validator expects (187 floats).
"""
import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt

from xr_hand.parser import parse_hand_message
from xr_hand.receiver import OSCHandReceiver
from xr_hand.recorder import FrameRecorder
from xr_hand.validator import StreamMonitor, validate_raw_message
from xr_hand.viz3d import HandViewer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1",
                   help="Local interface to bind (use 0.0.0.0 to accept from any).")
    p.add_argument("--port", type=int, default=9002,
                   help="UDP port XR Trainer sends to (9002 in current beta).")
    p.add_argument("--address", default="/v1/animation/kinematic/all",
                   help="OSC address for the real-glove kinematic stream.")
    p.add_argument("--left-addr", default="/hand/left",
                   help="Legacy mock per-hand address (left).")
    p.add_argument("--right-addr", default="/hand/right",
                   help="Legacy mock per-hand address (right).")
    p.add_argument("--dump", action="store_true",
                   help="Print one log line per received packet.")
    p.add_argument("--no-viz", action="store_true",
                   help="Run validator only, no GUI window.")
    p.add_argument("--record", type=Path, default=None,
                   help="Also write each validated frame to this .jsonl file.")
    p.add_argument("--duration", type=float, default=None,
                   help="Auto-close the viewer after this many seconds.")
    p.add_argument("--hz", type=float, default=5.0,
                   help="Frames saved per second when --record is set "
                        "(default: 5). Use 0 to keep every frame.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("osc")

    receiver = OSCHandReceiver(
        host=args.host, port=args.port,
        kinematic_addr=args.address,
        left_addr=args.left_addr, right_addr=args.right_addr,
        dump=args.dump,
    )
    receiver.start()

    monitors = {"left": StreamMonitor("left"), "right": StreamMonitor("right")}

    def process(hand: str, raw: list) -> "tuple[bool, object]":
        result = validate_raw_message(raw)
        if not result.is_valid:
            log.warning("[%s] INVALID: %s", hand, "; ".join(result.errors))
            return False, None
        for w in result.warnings:
            log.warning("[%s] %s", hand, w)
        frame = parse_hand_message(raw, hand_side_hint=hand)
        for w in monitors[hand].update(frame.packet_counter):
            log.warning("[%s] %s", hand, w)
        return True, frame

    if args.no_viz:
        log.info("headless mode; Ctrl+C to stop")
        try:
            while True:
                for hand, raw in receiver.drain(max_items=64):
                    process(hand, raw)
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            receiver.stop()
            log.info("msg_count=%s dropped=%d", receiver.msg_count, receiver.dropped)
        return

    viewer = HandViewer(title=f"XR Hand (OSC {args.host}:{args.port})")

    recorder = None
    if args.record:
        recorder = FrameRecorder(hz=args.hz or None)
        recorder.start(args.record)
        log.info("recording to %s at %s", args.record,
                 f"{args.hz} Hz" if args.hz else "full rate")

    def tick() -> None:
        for hand, raw in receiver.drain(max_items=16):
            ok, frame = process(hand, raw)
            if ok:
                if frame.packet_counter % 60 == 0:
                    log.info("[%s] counter=%d", hand, frame.packet_counter)
                if recorder is not None:
                    recorder.record(frame)
                viewer.update_hand(frame)

    if args.duration is not None:
        timer = viewer.fig.canvas.new_timer(interval=int(args.duration * 1000))
        timer.add_callback(lambda: plt.close("all"))
        timer.start()

    try:
        viewer.run(tick, interval_ms=33)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        if recorder is not None:
            recorder.stop()
            log.info("saved %d frames to %s", recorder.count, args.record)
        log.info("msg_count=%s dropped=%d", receiver.msg_count, receiver.dropped)


if __name__ == "__main__":
    main()
