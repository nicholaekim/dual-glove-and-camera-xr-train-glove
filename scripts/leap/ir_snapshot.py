"""IR stills from the Stereo IR 170 — the professor-facing gate evidence.

The Phase 2 gate asks one question: does the IR camera see a hand inside the
black StretchSense glove? The stats table answers it in numbers, but a table
cannot show *why* a gloved hand was not tracked. These photographs can: they
are the raw 850 nm brightness images, one per eye, exactly what the tracker
was given.

Every snapshot writes three files:

    <label>_<k>_L.png    left eye, 8-bit greyscale
    <label>_<k>_R.png    right eye
    <label>_<k>.json     frame id, both clocks, image size, and the hands the
                         tracker reported in the nearest tracking event
                         (side, id, palm position in metres)

The sidecar is not optional bookkeeping. A still of a gloved hand is only
evidence if it also says whether the tracker saw that hand at that instant,
and "no hand tracked" next to a perfectly visible glove is exactly the
finding the gate is looking for.

Protocol (plan section 6): module flat on the table, lenses up, hand 20 to 50
cm above it, no sunlight and no other IR sources. "Allow images" must be on
in the Ultraleap Control Panel — if it is off the run stops with that as the
message rather than writing blank frames.

  python scripts/leap/ir_snapshot.py --label bare
  python scripts/leap/ir_snapshot.py --label glove --count 10 --interval 0.5
  python scripts/leap/ir_snapshot.py --out results/leap_gate/ir --label glove_tape
  python scripts/leap/ir_snapshot.py --mock --label demo     # synthetic gradients

--mock writes obviously synthetic gradient PNGs through the same sampler ->
numpy -> imwrite -> sidecar path, so the pipeline can be exercised on a
machine with no camera. Nothing it writes is evidence of anything.
"""
import argparse
import sys
import time
from pathlib import Path

from leap_hand.images import HandTrail, open_sampler, write_snapshot
from leap_hand.stream import LeapUnavailable, open_stream

DEFAULT_OUT = Path("results") / "leap_gate" / "ir"
MOCK_NOTE = "synthetic gradient from --mock: not camera data, not evidence"


def drain_into(trail: HandTrail, source) -> int:
    """Move every pending tracked hand into the trail. Returns how many."""
    hands = source.drain(64)
    for _side, lh in hands:
        trail.add(lh)
    return len(hands)


SETTLE_S = 0.4      # let tracking events flow before the first still


def capture(source, sampler, trail: HandTrail, out_dir: Path, label: str,
            count: int, interval: float, mock: bool) -> list:
    """Take `count` snapshots `interval` seconds apart. Returns the records."""
    # `open_stream` clears its queue on the way out, so without this the first
    # snapshot would be paired against an empty trail and report "no hand"
    # even with a hand over the module.
    t_end = time.time() + SETTLE_S
    while time.time() < t_end:
        drain_into(trail, source)
        time.sleep(0.005)

    snaps = []
    for k in range(count):
        if k:
            # Keep draining through the gap: the trail must be fresh when the
            # next image lands, or the sidecar pairs a photo with a stale hand.
            t_end = time.time() + interval
            while time.time() < t_end:
                drain_into(trail, source)
                time.sleep(0.005)
        drain_into(trail, source)

        pair = sampler.wait_for_pair(timeout=2.0)
        if pair is None:
            print(f"  [{k:03d}] no image within 2 s — is 'Allow images' on in "
                  "the Control Panel?")
            continue
        drain_into(trail, source)
        hands, dt_ms = trail.nearest(pair.timestamp_us)

        snap = write_snapshot(out_dir, label, k, pair, hands=hands,
                              tracking_dt_ms=dt_ms,
                              note=MOCK_NOTE if mock else "")
        snaps.append(snap)
        means = " ".join(f"{s}:{v:.1f}" for s, v in snap.mean_pixel.items())
        print(f"  [{k:03d}] {pair.size_text}  frame {pair.frame_id}  "
              f"mean {means}  {snap.tracking_text}")
    return snaps


def main() -> None:
    p = argparse.ArgumentParser(
        description="Save IR stills (both eyes) from the Ultraleap camera, "
                    "each with a sidecar saying what the tracker saw.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"folder for the PNGs and sidecars (default: {DEFAULT_OUT})")
    p.add_argument("--count", type=int, default=5,
                   help="snapshots to take (default: 5)")
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between snapshots (default: 1.0)")
    p.add_argument("--label", default="ir",
                   help="filename prefix, normally the gate condition "
                        "(default: ir)")
    p.add_argument("--mock", action="store_true",
                   help="synthetic gradient images; no camera or SDK needed")
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"))
    p.add_argument("--timeout", type=float, default=5.0,
                   help="seconds to wait for a device (default: 5)")
    args = p.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    print("=" * 62)
    print("Ultraleap IR snapshots" + ("  [mock]" if args.mock else ""))
    print(f"  {args.count} x {args.interval:g} s   label: {args.label}   "
          f"out: {args.out}")
    print("=" * 62)

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

    trail = HandTrail()
    snaps = []
    try:
        print(f"device: {getattr(source, 'device_serial', None) or 'unknown'}\n")
        snaps = capture(source, sampler, trail, args.out, args.label,
                        args.count, args.interval, args.mock)
    except KeyboardInterrupt:
        print("\nInterrupted — keeping the snapshots taken so far.")
    finally:
        sampler.detach(getattr(source, "connection", None))
        source.stop()

    print()
    if not snaps:
        print("No snapshots written.")
        sys.exit(2)
    with_hand = sum(1 for s in snaps if s.saw_hand)
    first = snaps[0].pair
    print(f"{len(snaps)} snapshots in {args.out}")
    print(f"  image size: {first.width} x {first.height}, {first.bpp} "
          f"byte(s) per pixel")
    print(f"  tracker reported a hand in {with_hand}/{len(snaps)}"
          + ("  <- no hand seen: that is itself the gate result"
             if not with_hand else ""))
    if getattr(sampler, "errors", 0):
        print(f"  {sampler.errors} image events could not be read "
              f"(last: {sampler.last_error})")


if __name__ == "__main__":
    main()
