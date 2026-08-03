"""Score the glove and the camera on the same poses, with the same features.

The point this settles: the July glove result was 90% with `pinch` classified
as `open_palm`, explained as stretch sensors being unable to sense thumb
opposition. That was an argument, not a measurement. This measures it.

Both pipelines already emit 21 wrist-centred keypoints in the identical
MediaPipe order (xr_hand.keypoints21 and cam_hand.export21), so one feature
function runs on both and the numbers are directly comparable. Each sensor is
scored twice:

  flexion only     the 5 wrist-to-fingertip distances — all a stretch sensor
                   can physically measure
  flexion+spread   plus adjacent-fingertip gaps, thumb-to-pinky-base, and the
                   thumb's offset out of the palm plane

Reading the four cells:

  * glove flexion vs glove flexion+spread — if adding spread does NOT help the
    glove, its 21 keypoints carry no real abduction information and the extra
    columns are reconstruction, not measurement. That is the claim behind the
    whole complementary-sensor argument, and this is what tests it.
  * camera flexion vs camera flexion+spread — whether measuring spread fixes
    the poses flexion alone confuses.
  * glove vs camera on the same feature set — like for like.

No simultaneous recording is needed. The two datasets are separate sessions of
the same pose set, which is all a pose-level comparison requires — and it is
the only option available, since the camera cannot see the hand while the
glove is worn (results/detection_tuning.csv).

  python scripts/compare_sensors.py
  python scripts/compare_sensors.py --glove <dir> --cam <dir> --write
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from cam_hand.export21 import wrist_centered
from cam_hand.features import (
    ALL_COLS,
    ALL_NAMES,
    FLEXION_COLS,
    all_features,
    loo_nearest_centroid,
    mean_vector,
)
from cam_hand.recorder import CamRecorder

HERE = Path(__file__).resolve().parents[1]
DEFAULT_GLOVE = HERE / "recordings" / "poses"
DEFAULT_CAM = HERE / "recordings" / "poses_cam"


def glove_samples(root: Path):
    """Glove recordings -> [(pose, hand, file, mean feature vector)]."""
    from xr_hand.keypoints21 import frame_to_keypoints21
    from xr_hand.recorder import FrameRecorder

    out = []
    for path in sorted(root.rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            labels = [json.loads(line) for line in f if line.strip()]
        per_hand = defaultdict(list)
        poses = {}
        for d, (frame, _wall) in zip(labels, FrameRecorder.load(path)):
            hand = frame.hand_side
            pts = frame_to_keypoints21(frame)
            per_hand[hand].append(all_features(pts, hand_side=hand))
            poses[hand] = d.get("pose", "")
        for hand, feats in sorted(per_hand.items()):
            if feats and poses[hand]:
                out.append((poses[hand], hand, path.name, mean_vector(feats)))
    return out


def cam_samples(root: Path):
    """Camera recordings -> the same shape."""
    out = []
    for path in sorted(root.rglob("*.jsonl")):
        per_hand = defaultdict(list)
        poses = {}
        for d in CamRecorder.load(path):
            if "world" not in d:
                continue
            hand = d["hand_side"]
            per_hand[hand].append(
                all_features(wrist_centered(d), hand_side=hand))
            poses[hand] = d.get("pose", "")
        for hand, feats in sorted(per_hand.items()):
            if feats and poses[hand]:
                out.append((poses[hand], hand, path.name, mean_vector(feats)))
    return out


def score(samples, cols):
    if len(samples) < 3:
        return None
    ok, n, wrong = loo_nearest_centroid([(s[0], s[3]) for s in samples], cols)
    return ok, n, wrong


def misses(samples, wrong):
    return [f"{samples[i][0]} ({samples[i][1]}) -> {got}"
            for _true, got, i in wrong]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Glove vs camera on the same poses and the same features.")
    p.add_argument("--glove", type=Path, default=DEFAULT_GLOVE)
    p.add_argument("--cam", type=Path, default=DEFAULT_CAM)
    p.add_argument("--common-only", action="store_true", default=True,
                   help="restrict to poses present in both datasets (default)")
    p.add_argument("--all-poses", dest="common_only", action="store_false")
    p.add_argument("--write", action="store_true",
                   help="write results/sensor_comparison.txt")
    args = p.parse_args()

    have_glove = args.glove.is_dir()
    have_cam = args.cam.is_dir()
    if not have_glove and not have_cam:
        raise SystemExit(f"no recordings found at {args.glove} or {args.cam}")

    g = glove_samples(args.glove) if have_glove else []
    c = cam_samples(args.cam) if have_cam else []

    if args.common_only and g and c:
        shared = {s[0] for s in g} & {s[0] for s in c}
        g = [s for s in g if s[0] in shared]
        c = [s for s in c if s[0] in shared]

    lines = ["=" * 72,
             "Glove vs camera - same poses, same features, same classifier",
             "=" * 72, ""]
    lines.append(f"glove  {args.glove}")
    lines.append(f"       {len(g)} samples (take x hand), "
                 f"{len({s[0] for s in g})} poses")
    lines.append(f"camera {args.cam}")
    lines.append(f"       {len(c)} samples (take x hand), "
                 f"{len({s[0] for s in c})} poses")
    if not c:
        lines.append("")
        lines.append("No camera recordings yet. Record the same pose set with:")
        lines.append("  python scripts/record_poses_cam.py --poses "
                     + ",".join(sorted({s[0] for s in g})) + " --takes 3")
    lines.append("")

    lines.append(f"{'sensor':<10}{'flexion only':>18}{'flexion+spread':>20}")
    lines.append("-" * 72)
    results = {}
    for name, samples in (("glove", g), ("camera", c)):
        if not samples:
            lines.append(f"{name:<10}{'(no data)':>18}{'(no data)':>20}")
            continue
        flex = score(samples, FLEXION_COLS)
        both = score(samples, ALL_COLS)
        results[name] = (samples, flex, both)
        cell = lambda r: f"{r[0]}/{r[1]} ({100.0 * r[0] / r[1]:.0f}%)" if r else "-"
        lines.append(f"{name:<10}{cell(flex):>18}{cell(both):>20}")

    for name, (samples, flex, both) in results.items():
        lines.append("")
        lines.append(f"{name} misclassifications")
        for label, r in (("flexion only", flex), ("flexion+spread", both)):
            if not r:
                continue
            m = misses(samples, r[2])
            lines.append(f"  {label:<16} " + (", ".join(m) if m else "none"))

    if "glove" in results and "camera" in results:
        lines.append("")
        lines.append("How to read it")
        gf, gb = results["glove"][1], results["glove"][2]
        if gf and gb:
            delta = gb[0] - gf[0]
            if delta <= 0:
                lines.append("  Adding spread does NOT help the glove "
                             f"({gf[0]} -> {gb[0]} correct). Its 21 keypoints")
                lines.append("  carry no usable abduction information: those "
                             "columns are reconstructed")
                lines.append("  by the hand model, not measured. This is the "
                             "measured form of the")
                lines.append("  'stretch sensors cannot see opposition' claim.")
            else:
                lines.append(f"  Adding spread helps the glove by {delta} "
                             "samples - so its keypoints do carry")
                lines.append("  some abduction signal. Worth checking whether "
                             "it is measured or inferred")
                lines.append("  from the vendor hand model before relying on it.")
        cf, cb = results["camera"][1], results["camera"][2]
        if cf and cb and cb[0] > cf[0]:
            lines.append(f"  Spread helps the camera by {cb[0] - cf[0]} samples "
                         "- it measures what the glove cannot.")
    lines.append("")
    lines.append(f"features: {', '.join(ALL_NAMES)}")

    report = "\n".join(lines)
    print(report)
    if args.write:
        out = HERE / "results" / "sensor_comparison.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
