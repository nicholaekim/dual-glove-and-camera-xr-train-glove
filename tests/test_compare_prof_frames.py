"""Our keypoint files against the professor's: palm fit, mirrored fit, scale."""
import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_prof_frames", ROOT / "scripts" / "leap" / "compare_prof_frames.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _hand() -> np.ndarray:
    """21 points in his layout, palm in z=0, fingers and thumb curling to -z."""
    pts = [[0.0, 0.0, 0.0]]
    thumb = [[-25, 10, -5], [-45, 30, -15], [-55, 45, -30], [-60, 55, -45]]
    pts += [list(map(float, p)) for p in thumb]
    for k, x in enumerate((-24.0, -8.0, 8.0, 24.0)):
        base_y = 45.0 - 3.0 * abs(k - 1.5)
        pts.append([x, base_y, 0.0])
        for j in range(1, 4):                                  # each finger curls a bit more
            pts.append([x * (1 + 0.05 * j), base_y + 22.0 * j - 3.0 * j * j, -12.0 * j * (1 + 0.2 * k)])
    return np.asarray(pts)


def _rot(deg: float, axis: int) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i, j = [a for a in range(3) if a != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _block(header: str, pts) -> str:
    lines = [header] + [f"{i} ({mod.LANDMARK_LABELS[i]}): ({x:.3f}, {y:.3f}, {z:.3f})"
                        for i, (x, y, z) in enumerate(pts)]
    return "\n".join(lines) + "\n"


def test_a_rotated_copy_fits_and_a_mirrored_one_is_flagged(tmp_path):
    his = _hand() + np.array([60.0, 380.0, 25.0])
    R = _rot(35, 0) @ _rot(-50, 1) @ _rot(20, 2)
    ours_same = (_hand() / 0.8) @ R.T + np.array([10.0, 200.0, -5.0])   # his = 0.8 x ours
    ours_mirror = ours_same * np.array([-1.0, 1.0, 1.0])

    ref, out = tmp_path / "ref", tmp_path / "out"
    out.mkdir()
    for fid, ours, label in (("153624", ours_same, "right"), ("156023", ours_mirror, "left")):
        d = ref / f"frame_{fid}_NA"
        d.mkdir(parents=True)
        (d / f"frame_{fid}.txt").write_text(_block(f"Frame {fid} | Hand ID 1499 (right)", his), encoding="utf-8")
        (out / f"frame_{fid}_keypoints.txt").write_text(
            _block(f"Frame 497 | Hand ID: {label} | Time: 2026-09-23T15:51:11.044", ours), encoding="utf-8")
    (out / "frame_999999_keypoints.txt").write_text(_block("Frame 1 | Hand ID: left", ours_same), encoding="utf-8")

    table = tmp_path / "t.csv"
    assert mod.main([str(out), "--reference", str(ref), "--csv", str(table)]) == 0
    rows = {r["frame"]: r for r in csv.DictReader(open(table, encoding="utf-8"))}
    assert set(rows) == {"153624", "156023"}                    # 999999 has no reference

    same = rows["153624"]
    assert float(same["rms"]) < 0.01 and same["fit"] == "same"
    assert abs(float(same["scale"]) - 0.8) < 1e-3
    assert same["his_label"] == "right" and same["our_label"] == "right"

    flipped = rows["156023"]
    assert float(flipped["rms_mirrored"]) < 0.01 and float(flipped["rms"]) > 5.0
    assert flipped["fit"] == "MIRROR" and float(flipped["ratio"]) < mod.CLEARLY_MIRRORED
    assert abs(float(flipped["scale"]) - 0.8) < 1e-3


def test_a_flat_hand_cannot_tell_a_mirror_from_a_rotation():
    """The documented limit: mirrored coplanar points are a rotation of them."""
    flat = _hand()
    flat[:, 2] = 0.0
    r = mod.compare(flat * np.array([-1.0, 1.0, 1.0]), flat)
    assert r["rms"] < 0.01 and r["rms_mirrored"] < 0.01
