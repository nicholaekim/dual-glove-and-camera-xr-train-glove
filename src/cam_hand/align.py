"""Rigid alignment between two sets of corresponding 3D points.

Used twice in this project:

  * comparing camera keypoints against the professor's tracker output, which
    lives in a different coordinate frame and (for the camera) a different
    scale;
  * fusing camera and glove skeletons, where the palm points are matched to
    solve the rotation between the two sensors' frames.

`umeyama` is the closed-form least-squares solution for rotation + uniform
scale + translation (Umeyama 1991, the scaled Kabsch algorithm). With
with_scale=False it reduces to plain Kabsch, i.e. rotation only.

Reflections are explicitly excluded: the determinant correction keeps the
result a proper rotation, so a left hand can never be "aligned" onto a right
hand by mirroring it.
"""
from typing import Sequence, Tuple

import numpy as np


def umeyama(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]],
            with_scale: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
    """Best (R, s, t) mapping src onto dst: dst ~= s * R @ src + t."""
    A = np.asarray(src, dtype=float)
    B = np.asarray(dst, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"need matching (N,3) arrays, got {A.shape} and {B.shape}")

    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - mu_a, B - mu_b

    cov = B0.T @ A0 / A.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0                      # forbid reflections
    R = U @ S @ Vt

    if with_scale:
        var_a = (A0 ** 2).sum() / A.shape[0]
        s = float((D * np.diag(S)).sum() / var_a) if var_a > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_b - s * R @ mu_a
    return R, s, t


def apply(R: np.ndarray, s: float, t: np.ndarray,
          pts: Sequence[Sequence[float]]) -> np.ndarray:
    P = np.asarray(pts, dtype=float)
    return (s * (R @ P.T)).T + t


def align_points(src, dst, with_scale: bool = True, subset=None):
    """Align src onto dst and report the fit.

    subset: indices used to SOLVE the transform (e.g. palm landmarks, which
    are near-rigid); the transform is then applied to all points and the
    error reported over all of them. None = solve on everything.

    Returns (aligned_src, rmse, per_point_error, scale).
    """
    A = np.asarray(src, dtype=float)
    B = np.asarray(dst, dtype=float)
    if subset is None:
        R, s, t = umeyama(A, B, with_scale)
    else:
        idx = list(subset)
        R, s, t = umeyama(A[idx], B[idx], with_scale)
    A2 = apply(R, s, t, A)
    err = np.linalg.norm(A2 - B, axis=1)
    return A2, float(np.sqrt((err ** 2).mean())), err, s
