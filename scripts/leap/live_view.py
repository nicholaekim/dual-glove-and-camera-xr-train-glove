"""Live 3D skeleton from the Stereo IR 170 — the first smoke test and the
tool for the Phase 2 gate.

Draws the converted `HandFrame` in the same viewer the glove uses
(`xr_hand.viz3d`), so a camera hand and a glove hand look alike on screen and
any mapping mistake shows up immediately: finger order, chirality, and
fingertips being farthest from the wrist. The HUD carries the numbers the
gate needs while you are wearing the glove and cannot type:

    89.7 Hz tracking      the rate LeapC reports, never assumed to be 90
    hands: left(2001) right(1001)
    re-acquisitions: 3    hand-id changes since start = losses recovered
    pinch 0.12 grab 0.98  per hand, from LeapC

What to check on hardware, palm open and facing the module (plan section 3):
finger order and chirality are right, fingertips are the farthest points from
the wrist, and the skeleton does not flicker or swap hands at the edges.

Usage:
  python scripts/leap/live_view.py                 # the camera
  python scripts/leap/live_view.py --mock          # no camera needed
  python scripts/leap/live_view.py --duration 30   # auto-close
  python scripts/leap/live_view.py --no-window     # headless: console only
"""
import argparse
import time


def main() -> None:
    p = argparse.ArgumentParser(description="Live 3D viewer for the Ultraleap camera.")
    p.add_argument("--mock", action="store_true",
                   help="synthetic hands; no camera or SDK needed")
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"),
                   help="tracking mode (default: desktop, the plan's protocol)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="auto-close after N seconds (0 = until the window closes)")
    p.add_argument("--no-window", action="store_true",
                   help="headless: print the HUD to the console instead")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="seconds to wait for a device (default: 5)")
    args = p.parse_args()

    from leap_hand.stream import LeapUnavailable, open_stream

    try:
        source = open_stream(mock=args.mock, mode=args.mode,
                             device_timeout=args.timeout)
    except LeapUnavailable as e:
        print(f"\nNo live tracking: {e}\n")
        raise SystemExit(1)

    print("Ultraleap live view" + ("  [mock]" if args.mock else ""))
    print(f"  device: {getattr(source, 'device_serial', None) or 'unknown'}"
          f"   mode: {args.mode}")
    print("  close the window (or Ctrl+C) to stop\n")

    state = HudState()
    try:
        if args.no_window:
            run_headless(source, state, args.duration)
        else:
            run_viewer(source, state, args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        print(f"\n{state.hands} hands in {state.elapsed():.1f} s, "
              f"{state.reacquisitions} re-acquisitions, "
              f"last tracking rate {state.framerate:.1f} Hz")


class HudState:
    """Everything the HUD shows, updated as hands arrive."""

    def __init__(self):
        self.t0 = time.time()
        self.hands = 0
        self.framerate = 0.0
        self.ids: dict = {}            # hand_side -> last hand_id
        self.reacquisitions = 0
        self.pinch: dict = {}
        self.grab: dict = {}
        self.last_seen: dict = {}

    def elapsed(self) -> float:
        return time.time() - self.t0

    def update(self, lh) -> None:
        self.hands += 1
        if lh.framerate:
            self.framerate = lh.framerate
        previous = self.ids.get(lh.hand_side)
        if previous is not None and previous != lh.hand_id:
            self.reacquisitions += 1
        self.ids[lh.hand_side] = lh.hand_id
        self.pinch[lh.hand_side] = lh.pinch_strength
        self.grab[lh.hand_side] = lh.grab_strength
        self.last_seen[lh.hand_side] = time.time()

    def lines(self) -> list:
        now = time.time()
        live = [s for s, t in self.last_seen.items() if now - t < 0.5]
        hands = "  ".join(f"{s}({self.ids[s]})" for s in sorted(live)) or "none"
        grip = "  ".join(
            f"{s}: pinch {self.pinch.get(s, 0.0):.2f} grab {self.grab.get(s, 0.0):.2f}"
            for s in sorted(live)
        )
        return [
            f"{self.framerate:5.1f} Hz tracking      {self.elapsed():5.1f} s",
            f"hands: {hands}",
            f"re-acquisitions: {self.reacquisitions}",
            grip or "pinch/grab: -",
        ]


def run_headless(source, state: HudState, duration: float) -> None:
    """No matplotlib: print the same numbers once a second."""
    next_report = time.time() + 1.0
    while True:
        for _side, lh in source.drain(64):
            state.update(lh)
        now = time.time()
        if now >= next_report:
            print("  " + " | ".join(state.lines()))
            next_report += 1.0
        if duration and state.elapsed() >= duration:
            return
        time.sleep(0.005)


def run_viewer(source, state: HudState, duration: float) -> None:
    """The 3D skeleton, in the glove's own viewer."""
    import matplotlib.pyplot as plt

    from leap_hand.to_openxr import to_hand_frame
    from xr_hand.viz3d import HandViewer

    viewer = HandViewer(title="Ultraleap live view")
    # viz3d has no HUD of its own, and this script must not change it: add the
    # text to the axes from out here instead.
    hud = viewer.ax.text2D(
        0.02, 0.98, "", transform=viewer.ax.transAxes, color="#7fd7ff",
        family="monospace", fontsize=9, va="top", ha="left",
    )

    if duration:
        timer = viewer.fig.canvas.new_timer(interval=int(duration * 1000))
        timer.add_callback(lambda: plt.close(viewer.fig))
        timer.single_shot = True
        timer.start()

    def tick() -> None:
        for _side, lh in source.drain(64):
            state.update(lh)
            viewer.update_hand(to_hand_frame(lh))
        hud.set_text("\n".join(state.lines()))

    viewer.run(tick, interval_ms=33)


if __name__ == "__main__":
    main()
