"""Forward kinematics: convert per-joint local transforms to world positions.

Real-glove streams (and skeletal animation in general) express each joint's
pose as a translation + rotation relative to its parent joint. To draw the
hand we chain those transforms through the bone hierarchy.

Convention: each joint's `(x, y, z)` is a translation in its parent's local
frame, and the rotation quaternion is XYZW (qx, qy, qz, qw).

`absolute_to_relative` is the inverse direction, for sources that measure
every joint in one world frame instead (a camera: see `leap_hand`). Feed its
output back through `forward_kinematics` and the world positions come back
unchanged, so camera data reaches every existing tool in the glove's own
convention.
"""
from typing import List, Sequence, Tuple

import numpy as np

from .joints import BONES, JOINT_NAMES, HandFrame, Joint

# child_idx -> parent_idx
PARENT = {child: parent for parent, child in BONES}


def quat_to_mat3(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert an XYZW quaternion to a 3x3 rotation matrix.

    Handles non-unit quaternions by normalising the magnitude internally.
    """
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xs, ys, zs = qx * s, qy * s, qz * s
    wx, wy, wz = qw * xs, qw * ys, qw * zs
    xx, xy, xz = qx * xs, qx * ys, qx * zs
    yy, yz, zz = qy * ys, qy * zs, qz * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


def mat3_to_quat(m: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to an XYZW quaternion.

    Shepperd's method: pick the largest of the four possible denominators so
    the division never happens near zero (naive w-first extraction loses all
    precision at 180 degrees).
    """
    m = np.asarray(m, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return quat_normalize((qx, qy, qz, qw))


def quat_normalize(q: Sequence[float]) -> Tuple[float, float, float, float]:
    """Scale an XYZW quaternion to unit length (identity if it is degenerate)."""
    qx, qy, qz, qw = (float(v) for v in q)
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / n, qy / n, qz / n, qw / n)


def quat_conjugate(q: Sequence[float]) -> Tuple[float, float, float, float]:
    """Conjugate of an XYZW quaternion — the inverse, for unit quaternions."""
    qx, qy, qz, qw = (float(v) for v in q)
    return (-qx, -qy, -qz, qw)


def quat_mul(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
    """Hamilton product of two XYZW quaternions: rotate by `b`, then by `a`.

    Matches the matrix convention here, i.e.
    `quat_to_mat3(*quat_mul(a, b)) == quat_to_mat3(*a) @ quat_to_mat3(*b)`.
    """
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def forward_kinematics_full(frame: HandFrame) -> Tuple[list, list]:
    """World positions *and* world rotations for every joint of a HandFrame.

    Returns `(positions, rotations)`:
      positions  [(x, y, z), ...] in JOINT_NAMES order — identical to
                 `forward_kinematics(frame)`.
      rotations  [3x3 numpy array, ...], each the joint's orientation in the
                 world frame (columns are the joint's local axes). Use
                 `mat3_to_quat` if you want XYZW quaternions instead.

    The rotations are what `forward_kinematics` computes internally and then
    throws away; they are what `absolute_to_relative` needs to go back.
    """
    n = len(frame.joints)
    world_pos = [None] * n
    world_rot = [None] * n

    def compute(idx: int) -> None:
        if world_pos[idx] is not None:
            return
        j = frame.joints[idx]
        local_rot = quat_to_mat3(j.qx, j.qy, j.qz, j.qw)
        local_pos = np.array([j.x, j.y, j.z])

        if idx not in PARENT:
            world_rot[idx] = local_rot
            world_pos[idx] = local_pos
        else:
            par = PARENT[idx]
            compute(par)
            world_rot[idx] = world_rot[par] @ local_rot
            world_pos[idx] = world_pos[par] + world_rot[par] @ local_pos

    for i in range(n):
        compute(i)

    positions = [(float(p[0]), float(p[1]), float(p[2])) for p in world_pos]
    return positions, world_rot


def forward_kinematics(frame: HandFrame) -> list:
    """Compute world positions for all 26 joints from a HandFrame.

    Returns [(x, y, z), ...] in JOINT_NAMES order.
    """
    return forward_kinematics_full(frame)[0]


def absolute_to_relative(
    abs_pos: Sequence[Sequence[float]],
    abs_quat_xyzw: Sequence[Sequence[float]],
    names: Sequence[str] = JOINT_NAMES,
) -> List[Joint]:
    """World-space joint poses -> the parent-relative Joints a HandFrame holds.

    The inverse of `forward_kinematics`. A camera measures every joint in one
    world frame; the glove convention (and therefore everything downstream of
    `HandFrame`) is parent-relative. So for each joint:

        root (WRIST)  keeps its absolute pose — it carries the hand's place
                      and orientation in the world, which is exactly the
                      information the glove never had.
        child         rel_pos  = R_parent^T (p_child - p_parent)
                      rel_quat = q_parent^-1 * q_child     (both normalised)

    `PARENT` comes from `BONES`, so WRIST is the root and PALM is its child.

    Units pass through untouched: feed metres, get metres.

    Args:
        abs_pos: 26 world positions [x, y, z], in `names` order.
        abs_quat_xyzw: 26 world rotations as XYZW quaternions, same order.
        names: joint names to stamp on the result (default JOINT_NAMES).

    Returns:
        A list of `Joint`, ready for `HandFrame(joints=...)`.
    """
    n = len(names)
    if len(abs_pos) != n or len(abs_quat_xyzw) != n:
        raise ValueError(
            f"expected {n} positions and {n} quaternions, "
            f"got {len(abs_pos)} and {len(abs_quat_xyzw)}"
        )

    quats = [quat_normalize(q) for q in abs_quat_xyzw]
    mats = [quat_to_mat3(*q) for q in quats]
    pos = [np.asarray(p, dtype=float) for p in abs_pos]

    joints: List[Joint] = []
    for i in range(n):
        if i not in PARENT:
            local = pos[i]
            qx, qy, qz, qw = quats[i]
        else:
            par = PARENT[i]
            local = mats[par].T @ (pos[i] - pos[par])
            qx, qy, qz, qw = quat_normalize(
                quat_mul(quat_conjugate(quats[par]), quats[i])
            )
        joints.append(Joint(
            name=names[i],
            x=float(local[0]), y=float(local[1]), z=float(local[2]),
            qw=float(qw), qx=float(qx), qy=float(qy), qz=float(qz),
        ))
    return joints
