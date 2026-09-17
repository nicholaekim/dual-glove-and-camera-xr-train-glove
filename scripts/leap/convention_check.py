"""Hardware convention check: is LeapC's geometry what the plan says it is?

`docs/ultraleap_ir170_plan.md` section 1 documents LeapC's conventions from
the struct reference, and `src/leap_hand/to_openxr.py` maps 26 joints on the
strength of them. This script is the measurement that closes that loop: it
holds a real hand in front of a real device and checks every claim
numerically, so a convention mistake is caught before any data is collected
rather than after.

What it checks, and what the plan predicts:

  bone axis        `rotation * (0,0,-1)` must point from `prev_joint` to
                   `next_joint` — a bone points along its own local -z.
                   Expect ~0 degrees.
  thumb metacarpal `digits[0].bones[0]` has zero length, which is why the
                   thumb chain in the mapping table starts at `bones[1]`.
  palm basis       `orientation` is {normal x direction, -normal, -direction},
                   so `orientation * (0,-1,0)` is `palm.normal` and
                   `orientation * (0,0,-1)` is `palm.direction`. Expect ~0.
  hand layout      fingertips are farther from the wrist than the knuckles,
                   which catches a mirrored or reversed chain.
  frame age        `leap.get_now() - event.timestamp` at receipt, reported
                   for the record (not a pass/fail).

Exit codes:
  0  a hand was measured and every check is within tolerance
  2  no hand appeared within --wait seconds (nothing was measured)
  3  a hand was measured and at least one check failed — do not collect data
     until the mapping in `to_openxr.py` is corrected
  1  the bindings or the tracking service are missing (the message says how)

  python scripts/leap/convention_check.py
  python scripts/leap/convention_check.py --wait 120 --sample 10

Hold one hand (or both) 25 to 40 cm above the module, palm down, and keep it
reasonably still. Recorded result of the first run on this unit: all zeros —
see plan section 9, "Hardware day".
"""
import argparse
import math
import sys
import threading
import time

import numpy as np

from leap_hand.stream import LeapStream, LeapUnavailable
from leap_hand.to_openxr import hand_side

# Tolerances. The conventions are exact statements about the data, not
# estimates, so anything above about a degree means the convention is wrong
# rather than noisy.
MAX_BONE_ANGLE_DEG = 5.0
MAX_PALM_ANGLE_DEG = 5.0
MAX_THUMB_METACARPAL_MM = 0.5

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def q_rot(q, v):
    """Rotate `v` by the unit quaternion `q` (a LeapC Quaternion: x,y,z,w)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    qv = np.array([x, y, z])
    v = np.asarray(v, float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def vec(p) -> np.ndarray:
    """A LeapC Vector -> a numpy array, still in millimetres."""
    return np.array([p.x, p.y, p.z], float)


def ang(a, b) -> float:
    """The angle between two vectors, in degrees."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


class Probe:
    """Collects per-hand measurements off the SDK's polling thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.samples = []
        self.frames = 0
        self.first_hand_t = None
        self.fps = []
        self.age_ms = []

    def on_tracking_event(self, event, now_us: int) -> None:
        with self.lock:
            self.frames += 1
            self.fps.append(event.framerate)
            if not event.hands:
                return
            if self.first_hand_t is None:
                self.first_hand_t = time.time()
            self.age_ms.append((now_us - event.timestamp) / 1000.0)
            for hand in event.hands:
                self.samples.append(self._measure(hand))

    @staticmethod
    def _measure(hand) -> dict:
        wrist = vec(hand.arm.next_joint)
        palm = vec(hand.palm.position)
        bone_angles = []
        bone_len = []
        for digit_index, digit in enumerate(hand.digits):
            for bone_index, bone in enumerate(digit.bones):
                p0, p1 = vec(bone.prev_joint), vec(bone.next_joint)
                length = float(np.linalg.norm(p1 - p0))
                bone_len.append((digit_index, bone_index, length))
                if length > 1e-6:
                    bone_angles.append(ang(q_rot(bone.rotation, [0, 0, -1]),
                                           p1 - p0))
        tips = [float(np.linalg.norm(vec(d.bones[3].next_joint) - wrist))
                for d in hand.digits]
        knuckles = [float(np.linalg.norm(vec(d.bones[1].prev_joint) - wrist))
                    for d in hand.digits]
        return dict(
            side=hand_side(hand), id=hand.id, vis=hand.visible_time / 1e6,
            bone_angle_mean=float(np.mean(bone_angles)),
            bone_angle_max=float(np.max(bone_angles)),
            thumb_meta_len=bone_len[0][2],
            palm_minus_wrist=float(np.linalg.norm(palm - wrist)),
            tips=tips, knuckles=knuckles,
            normal_err=ang(q_rot(hand.palm.orientation, [0, -1, 0]),
                           vec(hand.palm.normal)),
            dir_err=ang(q_rot(hand.palm.orientation, [0, 0, -1]),
                        vec(hand.palm.direction)),
            palm_y=float(palm[1]),
            pinch=hand.pinch_strength, grab=hand.grab_strength,
        )


def collect(probe: Probe, wait_s: float, sample_s: float) -> None:
    """Connect, wait for a hand, sample it for `sample_s`, disconnect."""
    leap = LeapStream.import_leap()

    class _Listener(leap.Listener):
        def on_tracking_event(self, event):
            probe.on_tracking_event(event, leap.get_now())

    connection = leap.Connection(listeners=[_Listener()])
    try:
        connection.connect()
        connection.set_tracking_mode(leap.enums.TrackingMode.Desktop)
    except Exception as e:
        try:
            connection.disconnect()
        except Exception:
            pass
        raise LeapUnavailable(
            f"could not start desktop tracking ({type(e).__name__}: {e}).\n"
            "  Next step: open the Ultraleap Control Panel, confirm the "
            "service is running and the device is listed, then:\n"
            "  python scripts/leap/check_setup.py"
        ) from e

    print(f"waiting up to {wait_s:.0f} s for a hand over the module ...",
          flush=True)
    try:
        t0 = time.time()
        while time.time() - t0 < wait_s:
            time.sleep(0.25)
            with probe.lock:
                started = probe.first_hand_t
            if started is not None and time.time() - started >= sample_s:
                break
    finally:
        connection.disconnect()


def report(probe: Probe) -> int:
    """Print the measurements and return the exit code."""
    with probe.lock:
        samples = list(probe.samples)
        frames, fps, age = probe.frames, list(probe.fps), list(probe.age_ms)

    mean_fps = float(np.mean(fps)) if fps else 0.0
    print(f"tracking frames: {frames}  mean fps {mean_fps:.1f}")
    if not samples:
        print("NO HAND SEEN. Hold a hand 25-40 cm above the module, palm "
              "down, and rerun.")
        return 2

    sides = {}
    for s in samples:
        sides[s["side"]] = sides.get(s["side"], 0) + 1
    print(f"hand samples: {len(samples)}  sides: {sides}  "
          f"ids: {sorted(set(s['id'] for s in samples))}")
    print(f"frame age at receipt: median {np.median(age):.1f} ms, "
          f"p95 {np.percentile(age, 95):.1f} ms")

    bone_mean = float(np.mean([s["bone_angle_mean"] for s in samples]))
    bone_worst = float(np.max([s["bone_angle_max"] for s in samples]))
    print(f"bone axis check  rotation*(0,0,-1) vs prev->next: "
          f"mean {bone_mean:.2f} deg, worst {bone_worst:.2f} deg  (expect ~0)")

    thumb_len = float(np.mean([s["thumb_meta_len"] for s in samples]))
    print(f"thumb bones[0] length: {thumb_len:.3f} mm (expect 0)")

    palm_wrist = float(np.mean([s["palm_minus_wrist"] for s in samples]))
    print(f"palm to wrist distance: {palm_wrist:.1f} mm (expect ~50-80)")

    normal_err = float(np.mean([s["normal_err"] for s in samples]))
    dir_err = float(np.mean([s["dir_err"] for s in samples]))
    print(f"palm basis: orientation*(0,-1,0) vs normal {normal_err:.2f} deg; "
          f"orientation*(0,0,-1) vs direction {dir_err:.2f} deg (expect ~0)")

    tips = np.mean([s["tips"] for s in samples], axis=0)
    knuckles = np.mean([s["knuckles"] for s in samples], axis=0)
    print("wrist->knuckle / wrist->tip (mm): " + ", ".join(
        f"{name} {k:.0f}/{t:.0f}"
        for name, k, t in zip(FINGER_NAMES, knuckles, tips)))
    tips_ok = bool(np.all(tips > knuckles))
    print(f"tips farther than knuckles for all fingers: {tips_ok}")

    print(f"palm height above module: "
          f"{np.mean([s['palm_y'] for s in samples]):.0f} mm;  "
          f"pinch {np.mean([s['pinch'] for s in samples]):.2f}  "
          f"grab {np.mean([s['grab'] for s in samples]):.2f}")

    failures = []
    if bone_worst > MAX_BONE_ANGLE_DEG:
        failures.append(f"bone axis off by {bone_worst:.2f} deg "
                        f"(tolerance {MAX_BONE_ANGLE_DEG:g})")
    if thumb_len > MAX_THUMB_METACARPAL_MM:
        failures.append(f"thumb bones[0] is {thumb_len:.3f} mm long, not zero")
    if max(normal_err, dir_err) > MAX_PALM_ANGLE_DEG:
        failures.append(f"palm basis off by {max(normal_err, dir_err):.2f} deg "
                        f"(tolerance {MAX_PALM_ANGLE_DEG:g})")
    if not tips_ok:
        failures.append("a fingertip is closer to the wrist than its knuckle "
                        "— the finger chain is reversed or mirrored")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print("\nThe mapping in src/leap_hand/to_openxr.py assumes these "
              "conventions (plan section 1).\nDo not collect data until this "
              "is resolved.")
        return 3
    print("PASS: every documented LeapC convention holds on this device "
          "(plan section 1).")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check LeapC's documented geometry conventions against a "
                    "real hand on a real device.")
    p.add_argument("--wait", type=float, default=60.0,
                   help="seconds to wait for a hand to appear (default: 60)")
    p.add_argument("--sample", type=float, default=5.0,
                   help="seconds to sample once a hand appears (default: 5)")
    args = p.parse_args()

    probe = Probe()
    try:
        collect(probe, args.wait, args.sample)
    except LeapUnavailable as e:
        print(f"\nNo live tracking: {e}\n")
        sys.exit(1)
    sys.exit(report(probe))


if __name__ == "__main__":
    main()
