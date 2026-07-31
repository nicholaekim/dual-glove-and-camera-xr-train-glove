"""Generate a SYNTHETIC simultaneous glove+camera dataset and check the
whole pipeline end to end — no glove, no webcam, no hardware at all.

Why this exists: the fusion claim is that some poses differ only in finger
spread or thumb opposition, which the glove physically cannot sense, so
glove-only classification must fail on them while camera and fused succeed.
Here that situation is constructed on purpose, with the answer known in
advance, so the machinery can be verified before a real session:

  fist / open_palm        differ in CURL      -> the glove can see these
  spread_v / together_v   differ only in SPREAD (same curl)
  pinch / relaxed         differ only in THUMB OPPOSITION (same curl)

The glove side is generated in the real 187-value wire format and pushed
through the real parser, so file formats, forward kinematics, pairing and
export are all exercised exactly as in production. The camera side is the
same underlying hand plus spread, plus a little noise.

The numbers are only as honest as the simulation: this proves the code does
what it claims on data whose answer is known, NOT that a real glove and a
real webcam behave this way. That is what scripts/record_simultaneous.py and
a real session are for.

  python scripts/selftest_sync.py           # write recordings/selftest, then
  python scripts/fuse_poses.py recordings/selftest
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

from cam_hand.fusion import FINGER_CHAINS, palm_frame
from cam_hand.landmarks import CamHand
from cam_hand.recorder import CamRecorder, finalize_pose_name, pose_filename

from xr_hand.joints import JOINT_NAMES
from xr_hand.keypoints21 import frame_to_keypoints21
from xr_hand.mock import _REST, _rotx_quat
from xr_hand.parser import parse_hand_message
from xr_hand.recorder import FrameRecorder

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)  # XYZW
FINGER_PREFIX = {"thumb": "THUMB", "index": "INDEX", "middle": "MIDDLE",
                 "ring": "RING", "pinky": "LITTLE"}
PHALANX_SUFFIXES = ("_PROXIMAL", "_INTERMEDIATE", "_DISTAL", "_TIP")

# curl: 0 = extended, 1 = fully curled (what stretch sensors measure)
# spread: extra abduction in degrees per finger (invisible to the glove)
# thumb_opp: thumb rotation out of the palm plane, degrees (also invisible)
POSES = {
    "open_palm":  {"curl": [0.0, 0.0, 0.0, 0.0, 0.0], "spread": 12.0, "thumb_opp": 0.0},
    "fist":       {"curl": [0.9, 1.0, 1.0, 1.0, 1.0], "spread": 0.0, "thumb_opp": 0.0},
    "spread_v":   {"curl": [0.8, 0.0, 0.0, 1.0, 1.0], "spread": 18.0, "thumb_opp": 0.0},
    "together_v": {"curl": [0.8, 0.0, 0.0, 1.0, 1.0], "spread": 0.0, "thumb_opp": 0.0},
    "relaxed":    {"curl": [0.3, 0.3, 0.3, 0.3, 0.3], "spread": 6.0, "thumb_opp": 0.0},
    "pinch":      {"curl": [0.3, 0.3, 0.3, 0.3, 0.3], "spread": 6.0, "thumb_opp": 55.0},
}
# Which poses a flexion-only sensor genuinely cannot tell apart.
GLOVE_BLIND_PAIRS = [("spread_v", "together_v"), ("relaxed", "pinch")]


def glove_wire_values(curl, hand: str, counter: int) -> list:
    """Per-finger curl -> the 187-value packet the glove would send."""
    values = [float(counter), float(counter), 1.0, 0.0, 0.0]
    curl_by_prefix = {FINGER_PREFIX[f]: c for f, c in zip(FINGER_CHAINS, curl)}
    for name in JOINT_NAMES:
        local = _REST[name]
        is_phalanx = any(name.endswith(s) for s in PHALANX_SUFFIXES)
        if is_phalanx:
            prefix = name.split("_")[0]
            angle = -curl_by_prefix.get(prefix, 0.0) * math.pi / 4
            q = _rotx_quat(angle)
        else:
            q = IDENTITY_QUAT
        if name == "WRIST":
            x, y, z = 0.0, 0.0, 0.0
        elif name.endswith("_METACARPAL"):
            px, py, pz = local
            x, y, z = (-px if hand == "left" else px), py, pz
        else:
            x, y, z = local
        values.extend([x, y, z, *q])
    return values


def rotate_about(axis, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    a = np.asarray(axis, float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    th = math.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def camera_view(pts, spread_deg: float, thumb_opp_deg: float,
                noise_mm: float, rng) -> np.ndarray:
    """The 'truth' hand: the glove's curl plus the spread it cannot sense."""
    P = np.asarray(pts, dtype=float).copy()
    n, _x, _y = palm_frame(P)
    # fan the fingers out around the palm normal, thumb-side to pinky-side
    fan = {"thumb": -1.4, "index": -1.0, "middle": 0.0, "ring": 1.0, "pinky": 2.0}
    for finger, chain in FINGER_CHAINS.items():
        knuckle = P[chain[0]].copy()
        R = rotate_about(n, spread_deg * fan[finger])
        P[chain] = (R @ (P[chain] - knuckle).T).T + knuckle
    if thumb_opp_deg:
        # Opposition swings the thumb out of the palm plane and across toward
        # the little finger. Rotating about the palm's long axis does that,
        # but the sign that means "toward the pinky" is mirrored between the
        # two hands — so pick the direction by its anatomical effect rather
        # than hard-coding a sign that is only right for one hand.
        chain = FINGER_CHAINS["thumb"]
        knuckle = P[chain[0]].copy()
        axis = P[9] - P[0]                      # wrist -> middle knuckle
        target = P[17]                          # pinky knuckle
        best = None
        for sign in (1.0, -1.0):
            R = rotate_about(axis, sign * thumb_opp_deg)
            moved = (R @ (P[chain] - knuckle).T).T + knuckle
            d = float(np.linalg.norm(moved[-1] - target))
            if best is None or d < best[0]:
                best = (d, moved)
        P[chain] = best[1]
    if noise_mm:
        P += rng.normal(0.0, noise_mm / 1000.0, P.shape)
    return P - P[0]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write a synthetic paired glove+camera dataset.")
    p.add_argument("--out-dir", type=Path, default=Path("recordings") / "selftest")
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--frames", type=int, default=25, help="frames per take")
    p.add_argument("--hands", default="left,right")
    p.add_argument("--noise-mm", type=float, default=4.0,
                   help="camera landmark noise, mm (default 4)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    hands = [h.strip() for h in args.hands.split(",") if h.strip()]
    glove_dir, cam_dir = args.out_dir / "glove", args.out_dir / "cam"
    glove_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    n_takes = 0
    for pose, spec in POSES.items():
        for take in range(1, args.takes + 1):
            name = pose_filename(pose, take)
            grec = FrameRecorder(pose=pose, take=take)
            crec = CamRecorder(pose=pose, take=take)
            grec.start(glove_dir / name)
            crec.start(cam_dir / name)
            for _ in range(args.frames):
                for hand in hands:
                    counter += 1
                    # small per-frame tremor so takes are not identical
                    curl = [min(1.0, max(0.0, c + rng.normal(0, 0.02)))
                            for c in spec["curl"]]
                    values = glove_wire_values(curl, hand, counter)
                    frame = parse_hand_message(values, hand_side_hint=hand)
                    grec.record(frame)
                    pts = frame_to_keypoints21(frame)
                    cam_pts = camera_view(pts, spec["spread"], spec["thumb_opp"],
                                          args.noise_mm, rng)
                    img = [[320.0 + x * 1000.0, 240.0 - y * 1000.0, z]
                           for x, y, z in cam_pts]
                    crec.record(CamHand(hand_side=hand, score=0.97,
                                        img=img, world=cam_pts.tolist()),
                                ts_ms=int(time.monotonic() * 1000),
                                frame_size=(640, 480))
                time.sleep(0.001)   # keep wall-clock stamps strictly increasing
            grec.stop()
            crec.stop()
            finalize_pose_name(glove_dir / name, set(hands))
            finalize_pose_name(cam_dir / name, set(hands))
            n_takes += 1
        print(f"  {pose:<12} {args.takes} takes x {args.frames} frames x "
              f"{len(hands)} hand(s)  curl={spec['curl']} "
              f"spread={spec['spread']:g} deg thumb_opp={spec['thumb_opp']:g} deg")

    print(f"\nwrote {n_takes} paired takes -> {args.out_dir}")
    print("  glove sees curl only; camera sees curl + spread + thumb opposition.")
    print("  poses a flexion-only sensor cannot separate: "
          + ", ".join(f"{a} vs {b}" for a, b in GLOVE_BLIND_PAIRS))
    print(f"\nNow run:  python scripts/fuse_poses.py {args.out_dir}")
    print("Expect: glove-only misses the spread/opposition pairs; "
          "camera and fused get them.")


if __name__ == "__main__":
    main()
